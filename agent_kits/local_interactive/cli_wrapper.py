"""Local Interactive CLI — drive the pipeline step-by-step from a terminal.

Commands mirror the pipeline stages plus a composite ``run``:

    fetch-video     resolve a video URI to a local path
    extract-frames  sample frames (prints the count)
    run-inference   detect objects (prints detections)
    write-output    persist a schema-valid RunOutput (always-ask gated)
    run             fetch → extract → infer → summarise (+ optional --out)

Global flags: ``--dry-run`` (describe, never execute) and ``--yes`` (auto-confirm
gated actions non-interactively). Without ``--yes`` gated actions prompt on stdin.
All commands emit schema-valid JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from ..common import RunInput
from ..common._yaml import load_file  # noqa: F401 - kept for parity/imports in docs
from .agent_harness import GatedActionDenied, LocalAgentHarness


def _make_confirm(auto_yes: bool):
    def confirm(action: str, detail: str) -> bool:
        if auto_yes:
            return True
        if not sys.stdin or not sys.stdin.isatty():
            return False  # fail-closed when non-interactive and not --yes
        reply = input(f"[confirm] {action}: {detail} — proceed? [y/N] ").strip().lower()
        return reply in ("y", "yes")
    return confirm


def _harness(args: argparse.Namespace) -> LocalAgentHarness:
    mode = "dry_run" if args.dry_run else args.mode
    return LocalAgentHarness(mode=mode, confirm_fn=_make_confirm(args.yes))


def _run_input(args: argparse.Namespace) -> RunInput:
    flags = {"fps": args.fps}
    if getattr(args, "model_version", None):
        flags["model_version"] = args.model_version
    return RunInput(video_uri=args.video, sensor_id=args.sensor,
                    processing_flags=flags, agent_id="local-interactive")


def _emit(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def cmd_fetch_video(args: argparse.Namespace) -> int:
    harness = _harness(args)
    if harness.mode == "dry_run":
        _emit({"executed": False, "would_fetch": args.video})
        return 0
    path = harness.fetcher.fetch_video(args.video)
    _emit({"executed": True, "local_path": path})
    return 0


def cmd_extract_frames(args: argparse.Namespace) -> int:
    harness = _harness(args)
    _emit(harness.run_skill("extract_frames", _run_input(args)))
    return 0


def cmd_run_inference(args: argparse.Namespace) -> int:
    harness = _harness(args)
    _emit(harness.run_skill("detect_objects", _run_input(args)))
    return 0


def cmd_write_output(args: argparse.Namespace) -> int:
    harness = _harness(args)
    try:
        result = harness.run(_run_input(args), store_dest=args.out)
    except GatedActionDenied as exc:
        _emit({"executed": False, "error": str(exc)})
        return 2
    _emit(result)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    harness = _harness(args)
    try:
        result = harness.run(_run_input(args), store_dest=args.out)
    except GatedActionDenied as exc:
        _emit({"executed": False, "error": str(exc)})
        return 2
    _emit(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_kits.local_interactive.cli_wrapper",
        description="Local Interactive kit — drive the drone-video pipeline.")
    parser.add_argument("--dry-run", action="store_true",
                        help="describe actions without executing")
    parser.add_argument("--mode", default="confirm",
                        choices=["dry_run", "confirm", "auto"], help="autonomy mode")
    parser.add_argument("--yes", action="store_true",
                        help="auto-confirm always-ask actions (non-interactive)")

    def add_common(sp: argparse.ArgumentParser, *, out: bool = False) -> None:
        sp.add_argument("--video", required=True, help="video URI (file:// or path)")
        sp.add_argument("--fps", type=float, default=1.0, help="frames per second")
        sp.add_argument("--sensor", default=None, help="sensor id")
        sp.add_argument("--model-version", default=None, help="pinned model version tag")
        if out:
            sp.add_argument("--out", default=None, help="output dir/file (write-output)")

    sub = parser.add_subparsers(dest="command", required=True)
    add_common(sub.add_parser("fetch-video", help="resolve a video URI"))
    add_common(sub.add_parser("extract-frames", help="sample frames"))
    add_common(sub.add_parser("run-inference", help="detect objects"))
    add_common(sub.add_parser("write-output", help="persist RunOutput (gated)"), out=True)
    add_common(sub.add_parser("run", help="full pipeline (+ optional --out)"), out=True)

    handlers = {
        "fetch-video": cmd_fetch_video,
        "extract-frames": cmd_extract_frames,
        "run-inference": cmd_run_inference,
        "write-output": cmd_write_output,
        "run": cmd_run,
    }
    parser.set_defaults(_handlers=handlers)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "out"):
        args.out = None
    return args._handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
