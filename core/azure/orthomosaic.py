"""Nadir drone video -> OpenDroneMap textured model -> unreferenced orthomosaic JPEG.

Ported from the workspace reference script ``generate_orthomosaic.py`` into an importable,
side-effect-free module that backs the ``/orthomosaic/`` endpoint and the background job
started when a video is uploaded (see ``apps/videos/orthomosaic_jobs.py``). The pipeline is
unchanged: frame extraction with a verified cache -> ODM (Docker or native) reconstruction to
a textured 2.5D model -> explicitly sized orthophoto render -> RGB JPEG preview.

A working Docker (image ``opendronemap/odm``) or native ODM install is required. When it is
missing, :func:`build_orthomosaic` raises :class:`OrthomosaicUnavailable` before any heavy work,
so callers can record a clean failure and carry on. ``cv2``, ``rasterio`` and ``PIL`` are
imported lazily so importing this module stays cheap.

The result is **unreferenced**: requested pixels are not a measured GSD, and there is no
absolute position, scale or north guarantee.
"""
import hashlib
import io
import json
import logging
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger('orthomosaic')
VERSION = 3
ORTHOMOSAIC_BLOB = 'orthomosaic.jpg'


class OrthomosaicUnavailable(RuntimeError):
    """The ODM engine (Docker image / native install) or an export dependency is not available."""


def orthomosaic_blob_name(account_id, video_pk):
    """Blob path in the shared frame layout: ``{account}/images/{video}/orthomosaic.jpg``."""
    return f'{account_id}/images/{video_pk}/{ORTHOMOSAIC_BLOB}'


def save_json(path, obj):
    path=Path(path)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def positive(value, name, zero=False):
    if not math.isfinite(value) or (value<0 if zero else value<=0):
        raise ValueError(f'{name} must be finite and '+('nonnegative' if zero else 'positive'))


def extract_frames(video_path, output_dir, seconds_interval=1, *, max_frame_size=2560,
                   min_sharpness=0, start=0, end=None):
    """Retains original interface; CFR video only; no arbitrary fallback FPS."""
    positive(seconds_interval,'interval')
    positive(start,'start',True)
    positive(min_sharpness,'min_sharpness',True)
    if max_frame_size<64:
        raise ValueError('max_frame_size must be at least 64')
    if end is not None and (not math.isfinite(end) or end<=start):
        raise ValueError('end must be finite and greater than start')
    video=Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f'Local video not found: {video}. Download URL inputs first.')
    folder=Path(output_dir).expanduser().resolve()
    folder.mkdir(parents=True,exist_ok=True)
    stat=video.stat()
    spec=dict(version=VERSION,video=str(video),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
              interval=seconds_interval,max_frame_size=max_frame_size,
              min_sharpness=min_sharpness,start=start,end=end)
    manifest=folder/'frames_manifest.json'
    if manifest.is_file():
        old=load_json(manifest)
        if old.get('spec')!=spec:
            raise RuntimeError('Extraction settings/input/version changed. Use new --frames-dir and --work-dir.')
        rows=old.get('frames',[])
        paths=[folder/r['name'] for r in rows]
        if not rows or not all(p.is_file() and sha256(p)==r['sha256'] for p,r in zip(paths,rows)):
            raise RuntimeError('Cached extracted frames changed/missing; use a new --frames-dir.')
        LOG.info('Reusing %d verified frames',len(paths))
        return list(map(str,paths))
    if any(folder.iterdir()):
        raise RuntimeError(f'Refusing nonempty unmanaged frame directory: {folder}. Choose an empty --frames-dir.')
    import cv2
    cap=cv2.VideoCapture(str(video))
    rows=[]
    try:
        if not cap.isOpened():
            raise RuntimeError('OpenCV could not open the video')
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps<=0:
            raise RuntimeError('Invalid reported FPS; inspect/transcode source instead of guessing FPS.')
        reported=float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count=int(reported) if math.isfinite(reported) and reported>0 else 0
        index=0; next_sample=start
        while cap.grab():
            t=index/fps
            if end is not None and t>=end:
                break
            if t+1e-9>=next_sample:
                ok,frame=cap.retrieve()
                if not ok or frame is None:
                    raise RuntimeError(f'Cannot decode sampled frame {index}')
                next_sample=start+(math.floor((t-start)/seconds_interval+1e-9)+1)*seconds_interval
                h,w=frame.shape[:2]
                scale=min(1,max_frame_size/max(h,w))
                if scale<1:
                    frame=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
                score=float(cv2.Laplacian(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
                if score>=min_sharpness:
                    name=f'frame_{index:010d}.jpg'
                    target=folder/name
                    if not cv2.imwrite(str(target),frame,[cv2.IMWRITE_JPEG_QUALITY,95]):
                        raise RuntimeError(f'Cannot write frame: {target}')
                    rows.append(dict(name=name,seconds=t,frame_index=index,sharpness=score,sha256=sha256(target)))
                    if len(rows)%25==0:
                        LOG.info('Saved %d frames; video time %.1fs',len(rows),t)
            index+=1
        if end is None and count and index<count-max(5,round(fps)):
            raise RuntimeError(f'Decoding ended unexpectedly early: {index}/{count} frames')
        if not rows:
            raise RuntimeError('No frames extracted; check time range and blur threshold')
        save_json(manifest,dict(spec=spec,fps=fps,frames=rows))
        LOG.info('Extracted %d color frames',len(rows))
        return [str(folder/r['name']) for r in rows]
    finally:
        cap.release()


def run_logged(command, log_path):
    LOG.info('Running: %s',subprocess.list2cmdline(command))
    with open(log_path,'a',encoding='utf-8') as log:
        log.write('\nCOMMAND: '+json.dumps(command)+'\n');log.flush()
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in proc.stdout:
                log.write(line);log.flush()
            code=proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()
            raise
        if code:
            suffix='; possible memory exhaustion' if code==137 else ''
            raise RuntimeError(f'Process exited with {code}{suffix}. See {log_path}')


def resolve_engine(backend, image, odm_run, odm_python):
    if backend=='docker':
        docker=shutil.which('docker')
        if not docker:
            raise RuntimeError('Docker not found. Install/start Docker or select native backend.')
        result=subprocess.run([docker,'image','inspect',image,'--format','{{.Id}}'],
                              capture_output=True,text=True,encoding='utf-8',errors='replace')
        if result.returncode:
            raise RuntimeError(f'Cannot inspect local ODM image {image}. Start Docker and run docker pull {image}.\n{result.stderr}')
        image_id=result.stdout.strip()
        if not image_id.startswith('sha256:'):
            raise RuntimeError(f'Unexpected Docker image identifier: {image_id}')
        return dict(backend='docker',image_id=image_id,docker=docker)
    run=Path(odm_run).expanduser().resolve() if odm_run else None
    if not run or not run.is_file():
        raise RuntimeError('Native backend requires --odm-run /path/to/ODM/run.py')
    return dict(backend='native',run_py=str(run),run_sha256=sha256(run),python=odm_python or sys.executable)


def reconstruction_options(fast, concurrency, feature_quality, pc_quality, matcher_order):
    positive(concurrency,'concurrency');positive(matcher_order,'matcher_order',True)
    options=['--feature-type','sift','--matcher-type','flann','--matcher-order',str(matcher_order),
             '--feature-quality',feature_quality,'--pc-quality',pc_quality,
             '--max-concurrency',str(concurrency),'--skip-3dmodel','--end-with','mvs_texturing']
    if fast:
        options.append('--fast-orthophoto')
    return options


def odm_command(project, engine, options, rerun=False):
    project=Path(project)
    args=['--project-path','/datasets' if engine['backend']=='docker' else str(project.parent),project.name,*options]
    if rerun:
        args.append('--rerun-all')
    if engine['backend']=='docker':
        return [engine['docker'],'run','--rm','--pull=never','-v',f'{project.parent}:/datasets',engine['image_id'],*args]
    return [engine['python'],engine['run_py'],*args]


def inspect_model(path):
    lo=[float('inf')]*3;hi=[float('-inf')]*3
    vertices=faces=uv=0;libraries=[]
    with open(path,encoding='utf-8',errors='replace') as f:
        for line in f:
            fields=line.split()
            if not fields:
                continue
            if fields[0]=='v' and len(fields)>=4:
                xyz=list(map(float,fields[1:4]))
                if not all(math.isfinite(x) for x in xyz):
                    raise ValueError('Nonfinite mesh coordinate')
                vertices+=1
                for j in range(3):lo[j]=min(lo[j],xyz[j]);hi[j]=max(hi[j],xyz[j])
            elif fields[0]=='f':faces+=1
            elif fields[0]=='vt':uv+=1
            elif fields[0]=='mtllib':
                single=path.parent/line.strip()[len('mtllib'):].strip()
                libraries.extend([single] if single.is_file() else [path.parent/x for x in fields[1:]])
    if not vertices or not faces or not uv:
        raise ValueError('Retained mesh lacks vertices, faces or texture coordinates')
    if not libraries or any(not p.is_file() for p in libraries):
        raise ValueError('Missing OBJ material library')
    textures=[]
    for lib in libraries:
        for line in lib.read_text(encoding='utf-8',errors='replace').splitlines():
            fields=line.strip().split(maxsplit=1)
            if fields and fields[0].lower()=='map_kd' and len(fields)==2:
                tex=lib.parent/fields[1].strip().strip('"')
                if not tex.is_file():
                    raise ValueError(f'Missing texture (or unsupported material options): {tex}')
                textures.append(tex)
    if not textures:
        raise ValueError('No diffuse texture maps found')
    return dict(min_xyz=lo,max_xyz=hi,span_xy=[hi[0]-lo[0],hi[1]-lo[1]],
                vertices=vertices,faces=faces,texture_coordinates=uv,
                assets=list(dict.fromkeys([path,*libraries,*textures])),textures=textures)


def artifact_signature(info, project):
    return [dict(path=p.relative_to(project).as_posix(),sha256=sha256(p)) for p in info['assets']]


def render_plan(info, long_edge, max_megapixels=64):
    if not 256<=long_edge<=20000:
        raise ValueError('long_edge must be between 256 and 20000 pixels')
    positive(max_megapixels,'max_megapixels')
    sx,sy=info['span_xy']
    if not all(math.isfinite(x) and x>0 for x in (sx,sy)):
        raise ValueError('Invalid XY mesh extent; inspect reconstruction/orientation')
    density=long_edge/max(sx,sy)
    w,h=math.ceil(sx*density),math.ceil(sy*density)
    if w*h>max_megapixels*1_000_000:
        raise ValueError(f'Target is {w*h/1e6:.1f} MP, over --max-megapixels {max_megapixels}; lower size or explicitly raise limit')
    return dict(pixels_per_model_unit=density,estimated_width=w,estimated_height=h,
                target_long_edge=long_edge,note='Model-coordinate density, NOT calibrated cm/pixel')


RENDER_CODE=r"""
import json,os,subprocess,sys
code_dir,model,output,ppu,threads=sys.argv[1:]
sys.path.insert(0,code_dir)
from opendm import context
args=[context.odm_orthophoto_path,'-inputFiles',model,'-outputFile',output,
      '-resolution',ppu,'-verbose','-logFile',os.path.join(os.path.dirname(output),'renderer.log'),
      '-outputCornerFile',os.path.join(os.path.dirname(output),'corners.txt'),
      '--config','GDAL_CACHEMAX','256']
print('RENDERER COMMAND: '+json.dumps(args),flush=True)
env=os.environ.copy();env['OMP_NUM_THREADS']=threads
subprocess.run(args,check=True,env=env)
"""


def renderer_command(project, model, tif, plan, engine, threads, code_dir='/code'):
    density=format(plan['pixels_per_model_unit'],'.17g')
    if engine['backend']=='docker':
        mapped=lambda p:'/datasets/'+p.relative_to(project.parent).as_posix()
        startup=('if [ -f "$1/scripts/pixi-shell-hook" ]; then source "$1/scripts/pixi-shell-hook"; fi; '
                 'if [ -f "$1/venv/bin/activate" ]; then source "$1/venv/bin/activate"; fi; '
                 'exec python3 -c "$2" "${@:3}"')
        return [engine['docker'],'run','--rm','--pull=never','--entrypoint','/bin/bash',
                '-v',f'{project.parent}:/datasets',engine['image_id'],'-c',startup,
                'odm-render',code_dir,RENDER_CODE,code_dir,mapped(model),mapped(tif),density,str(threads)]
    return [engine['python'],'-c',RENDER_CODE,str(Path(engine['run_py']).parent),str(model),str(tif),density,str(threads)]


def reconstruction_summary(project):
    p=project/'opensfm'/'reconstruction.json'
    total=sum(1 for x in (project/'images').glob('*.jpg'))
    if not p.is_file():
        return dict(input_images=total,warning='reconstruction.json unavailable')
    models=load_json(p)
    if not isinstance(models,list):
        return dict(input_images=total,warning='Unexpected reconstruction format')
    registered=set().union(*(set(m.get('shots',{})) for m in models)) if models else set()
    return dict(input_images=total,components=len(models),registered_unique=len(registered),
                registered_per_component=[len(m.get('shots',{})) for m in models],
                registered_fraction=len(registered)/total if total else None)


def validate_raster(tif, target):
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(tif) as src:
        if src.count<3 or any(d!='uint8' for d in src.dtypes[:3]):
            raise RuntimeError('Expected uint8 RGB orthophoto from RGB video')
        if max(src.width,src.height)<max(256,target*.8):
            raise RuntimeError(f'Unexpectedly small raster {src.width}x{src.height}; target {target}')
        mask=src.dataset_mask(out_shape=(min(src.height,512),min(src.width,512)),resampling=Resampling.nearest)
        if not mask.any():
            raise RuntimeError('Rendered raster contains no valid pixels')
        return dict(width=src.width,height=src.height,bands=src.count,
                    warning='Unreferenced raster: no metric scale, absolute position or north guarantee')


def reconstruct(frame_paths, project_dir='orthomosaic_work/project', *, long_edge=6000,
                max_megapixels=64, backend='docker', odm_image='opendronemap/odm:latest',
                odm_run=None, odm_python=None, concurrency=4, fast=False,
                feature_quality='high', pc_quality='medium', matcher_order=12,
                rerun=False, container_code_dir='/code'):
    """Return a newly rendered, explicitly sized TIFF; output size is a render-only setting."""
    import rasterio  # noqa: F401  Fail before expensive reconstruction if export dependencies are absent.
    from PIL import Image
    import numpy
    render_plan({'span_xy':[1,1]},long_edge,max_megapixels=max(400,max_megapixels))
    positive(max_megapixels,'max_megapixels')
    paths=[Path(p).expanduser().resolve() for p in frame_paths]
    if len(paths)<5:
        raise ValueError('This wrapper requires at least five overlapping frames')
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError('Input frame missing')
    project=Path(project_dir).expanduser().resolve()
    options=reconstruction_options(fast,concurrency,feature_quality,pc_quality,matcher_order)
    engine=resolve_engine(backend,odm_image,odm_run,odm_python)
    # Executable location is not engine identity; image ID is immutable for Docker.
    identity={k:v for k,v in engine.items() if k!='docker'}
    rows=[dict(name=f'frame_{i:07d}.jpg',sha256=sha256(p)) for i,p in enumerate(paths)]
    spec=dict(version=VERSION,images=rows,engine=identity,options=options)
    manifest=project/'pipeline_manifest.json'
    if manifest.is_file():
        if load_json(manifest)!=spec:
            raise RuntimeError('Reconstruction inputs/settings/engine/version changed. Use a NEW --work-dir. Output size alone may change.')
        image_dir=project/'images'
        if not image_dir.is_dir() or {p.name for p in image_dir.iterdir()}!={r['name'] for r in rows}:
            raise RuntimeError('Staged image set modified; use a new --work-dir')
        if any(sha256(image_dir/r['name'])!=r['sha256'] for r in rows):
            raise RuntimeError('Staged frame bytes modified; use a new --work-dir')
    else:
        if project.exists() and any(project.iterdir()):
            raise RuntimeError('Unmanaged/older project: use a NEW --work-dir; no automatic cache migration')
        (project/'images').mkdir(parents=True,exist_ok=True)
        for src,row in zip(paths,rows):shutil.copyfile(src,project/'images'/row['name'])
        save_json(manifest,spec)
    model=project/'odm_texturing_25d'/'odm_textured_model_geo.obj'
    mesh_done=project/'mesh_success.json'
    render_done=project/'render_success.json'
    if rerun or not mesh_done.is_file():
        mesh_done.unlink(missing_ok=True);render_done.unlink(missing_ok=True)
        run_logged(odm_command(project,engine,options,rerun),project/'reconstruction.log')
        if not model.is_file():
            raise RuntimeError(f'ODM finished without the expected textured 2.5D model: {model}')
        info=inspect_model(model)
        save_json(mesh_done,dict(assets=artifact_signature(info,project)))
    else:
        if not model.is_file():
            raise RuntimeError('Cached model missing; restore it or use --rerun with the same settings')
        info=inspect_model(model)
        if load_json(mesh_done).get('assets')!=artifact_signature(info,project):
            raise RuntimeError('Model/texture assets changed; restore them or use --rerun')
        LOG.info('Reusing verified textured model; no feature/dense/mesh/texturing recomputation')
    plan=render_plan(info,long_edge,max_megapixels)
    summary=reconstruction_summary(project)
    if summary.get('components',0)>1:
        LOG.warning('Multiple reconstruction components: %s',summary)
    if summary.get('registered_fraction') is not None and summary['registered_fraction']<.8:
        LOG.warning('Fewer than 80%% of frames registered (heuristic warning): %s',summary)
    tif_dir=project/'odm_orthophoto';tif_dir.mkdir(exist_ok=True)
    tif=tif_dir/'odm_orthophoto.tif'
    render_spec=dict(version=VERSION,mesh_assets=load_json(mesh_done)['assets'],plan=plan,
                     renderer_engine=identity,container_code_dir=container_code_dir)
    if not (render_done.is_file() and tif.is_file() and load_json(render_done).get('spec')==render_spec
            and load_json(render_done).get('tif_sha256')==sha256(tif)):
        render_done.unlink(missing_ok=True)
        partial=tif_dir/'orthophoto_rendering.tif'
        partial.unlink(missing_ok=True)
        LOG.info('Rendering approximately %dx%d pixels at %.6g pixels/model-unit',
                 plan['estimated_width'],plan['estimated_height'],plan['pixels_per_model_unit'])
        run_logged(renderer_command(project,model,partial,plan,engine,concurrency,container_code_dir),tif_dir/'render_console.log')
        result=validate_raster(partial,long_edge)
        # Preserve previous final TIFF until the new render has passed validation.
        partial.replace(tif)
        save_json(render_done,dict(spec=render_spec,tif_sha256=sha256(tif),result=result))
    else:
        result=validate_raster(tif,long_edge)
        LOG.info('Reusing verified render')
    texture_sizes=[]
    for p in info['textures']:
        with Image.open(p) as im:texture_sizes.append(dict(file=str(p),width=im.width,height=im.height))
    save_json(project/'orthomosaic_report.json',dict(reconstruction=summary,model=str(model),
              bounds=dict(min_xyz=info['min_xyz'],max_xyz=info['max_xyz']),textures=texture_sizes,
              render_plan=plan,result=result,engine=identity,
              limitations='Requested pixels are not guaranteed recoverable detail, calibrated GSD, registration accuracy or ground coverage'))
    return tif


def read_preview(tif, max_size=None):
    """Original BGR-array return convention; no downsampling unless explicitly requested."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling,ColorInterp
    with rasterio.open(tif) as src:
        if max_size is not None:positive(max_size,'preview_max_size')
        scale=min(1,max_size/max(src.width,src.height)) if max_size else 1
        w,h=max(1,round(src.width*scale)),max(1,round(src.height*scale))
        rgb=src.read([1,2,3],out_shape=(3,h,w),resampling=Resampling.bilinear)
        if ColorInterp.alpha in src.colorinterp:
            index=src.colorinterp.index(ColorInterp.alpha)+1
            valid=src.read(index,out_shape=(h,w),resampling=Resampling.nearest)>0
        else:
            valid=src.dataset_mask(out_shape=(h,w),resampling=Resampling.nearest)>0
        if not valid.any():raise RuntimeError('No valid orthophoto pixels')
        rgb[:,~valid]=255
        LOG.info('Full TIFF: %dx%d; viewing image: %dx%d',src.width,src.height,w,h)
        return np.ascontiguousarray(np.moveaxis(rgb,0,-1)[:,:,::-1])


def stitch_frames(frame_paths, *, project_dir='orthomosaic_work/project', preview_max_size=None, **kwargs):
    return read_preview(reconstruct(frame_paths,project_dir,**kwargs),preview_max_size)


def default_work_dir(video_path):
    """Separate project per source path; keep pre-v3 project directories untouched."""
    video=Path(video_path).expanduser().resolve()
    slug=re.sub(r'[^A-Za-z0-9_.-]+','_',video.stem).strip('._')[:60] or 'video'
    key=hashlib.sha256(str(video).encode('utf-8')).hexdigest()[:10]
    return str(Path('orthomosaic_work')/(slug+'-'+key))


def build_orthomosaic(video_path, work_dir, *, long_edge=6000, max_megapixels=64, interval=1,
                      start=0, end=None, max_frame_size=2560, min_sharpness=0, backend='docker',
                      odm_image='opendronemap/odm:latest', odm_run=None, odm_python=None,
                      concurrency=4, fast=False, feature_quality='high', pc_quality='medium',
                      matcher_order=12, rerun=False, container_code_dir='/code',
                      preview_max_size=None):
    """Run the whole pipeline for one local video; return ``(jpeg_bytes, report)``.

    Equivalent to ``generate_orthomosaic.py <video> --work-dir <work_dir>`` followed by saving
    the preview as a 95-quality, 4:4:4 JPEG, but returns the bytes instead of writing a file.
    Raises :class:`OrthomosaicUnavailable` if the engine or ``rasterio`` is missing.
    """
    try:
        import rasterio  # noqa: F401
    except ImportError as exc:
        raise OrthomosaicUnavailable(f'rasterio is required to export the orthomosaic: {exc}') from exc
    if not 256 <= long_edge <= 20000:
        raise ValueError('long_edge must be between 256 and 20000')
    positive(max_megapixels, 'max_megapixels')
    positive(concurrency, 'concurrency')
    positive(matcher_order, 'matcher_order', True)
    if preview_max_size is not None:
        positive(preview_max_size, 'preview_max_size')
    work = Path(work_dir).expanduser().resolve()
    # Check the engine before decoding a large video; reconstruct() resolves it again for identity.
    try:
        resolve_engine(backend, odm_image, odm_run, odm_python)
    except RuntimeError as exc:
        raise OrthomosaicUnavailable(str(exc)) from exc
    frames = extract_frames(video_path, work / 'frames', interval, max_frame_size=max_frame_size,
                            min_sharpness=min_sharpness, start=start, end=end)
    tif = reconstruct(frames, work / 'project', long_edge=long_edge, max_megapixels=max_megapixels,
                      backend=backend, odm_image=odm_image, odm_run=odm_run, odm_python=odm_python,
                      concurrency=concurrency, fast=fast, feature_quality=feature_quality,
                      pc_quality=pc_quality, matcher_order=matcher_order, rerun=rerun,
                      container_code_dir=container_code_dir)
    bgr = read_preview(tif, preview_max_size)
    import numpy as np
    from PIL import Image
    image = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95, subsampling=0)
    report_path = tif.parent.parent / 'orthomosaic_report.json'
    report = load_json(report_path) if report_path.is_file() else {}
    report['output'] = dict(width=image.width, height=image.height, bytes=buffer.tell())
    return buffer.getvalue(), report
