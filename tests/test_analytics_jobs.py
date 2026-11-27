"""AnalyticsJob lifecycle: transitions, runners, cooperative cancel, slot waits, API."""

import datetime
import json
from unittest import mock

import pytest
from django.utils import timezone
from rest_framework.test import APIClient

from apps.jobs import runners, services
from apps.jobs.models import AnalyticsJob, set_progress, transition
from apps.jobs.tasks import process_drone_video_job
from apps.media.models import MediaAsset
from apps.users.models import User
from apps.videos import vehicles

S = AnalyticsJob.Status
REPORT = {"tracks": [
    {"id": 1, "class": "car", "first_t": 2.0, "last_t": 9.0, "dominant_state": "MOVING",
     "trajectory_ref_px": [[2.0, 640.5, 360.0], [3.0, 650.0, 362.0]],
     "uncertainty": {"mean_conf": 0.81}},
    {"id": 2, "class": "truck", "first_t": 0.5, "last_t": 4.0, "dominant_state": "STATIONARY",
     "trajectory_ref_px": [[0.5, 100.0, 200.0]], "uncertainty": {"mean_conf": 0.66}},
]}


@pytest.fixture
def user(db):
    return User.objects.create_user(email="ops@example.com", password="pw-123456")


@pytest.fixture
def client(user):
    c = APIClient()
    c.force_authenticate(user=user)
    return c


@pytest.fixture
def alerts():
    """Capture broadcasts instead of pushing them."""
    with mock.patch("apps.jobs.tasks.broadcast_live_alert") as task_alerts, \
            mock.patch("apps.notifications.broadcast.broadcast_live_alert"):
        yield task_alerts


def make_asset(user, **fields):
    asset = MediaAsset(owner=user, filename="clip.mp4", content_type="video/mp4",
                       file_size_bytes=10, storage_provider="azure", bucket="input",
                       upload_status=MediaAsset.UploadStatus.UPLOADED, uploaded_at=timezone.now())
    for key, value in fields.items():
        setattr(asset, key, value)
    asset.object_key = f"{user.pk}/uploads/{asset.flight_id or '_'}/{asset.id}/clip.mp4"
    asset.save()
    return asset


def make_job(user, asset, job_type="VEHICLE_TRACKING", **params):
    return services.create_job(user, asset, job_type, parameters=params, dispatch_now=False)


def completed_tracking(fake_store, account_id, video_key):
    fake_store.put(vehicles.blob_names(account_id, video_key)["report"], json.dumps(REPORT).encode())
    return {"state": vehicles.COMPLETED, "cached": False, "summary": {"vehicles": {"total": 2}}}


# ----------------------------------------------------------------- state machine
@pytest.mark.django_db
class TestTransitions:
    def test_legal_and_illegal_edges(self, user):
        job = make_job(user, make_asset(user))
        assert not transition(job.pk, S.COMPLETED)              # QUEUED -> COMPLETED is illegal
        assert transition(job.pk, S.PROCESSING)
        assert not transition(job.pk, S.PROCESSING)             # already owned
        assert transition(job.pk, S.COMPLETED)
        for target in (S.QUEUED, S.CANCELLED, S.FAILED, S.PROCESSING):
            assert not transition(job.pk, target)               # COMPLETED is final
        job.refresh_from_db()
        assert job.started_at and job.completed_at

    def test_cancel_wins_race_against_completion(self, user):
        job = make_job(user, make_asset(user))
        transition(job.pk, S.PROCESSING)
        assert transition(job.pk, S.CANCELLED, from_=(S.QUEUED, S.PROCESSING))
        assert not transition(job.pk, S.COMPLETED, from_=(S.PROCESSING,))
        assert AnalyticsJob.objects.get(pk=job.pk).status == S.CANCELLED

    def test_progress_is_monotonic_and_only_while_processing(self, user):
        job = make_job(user, make_asset(user))
        assert not set_progress(job.pk, 50)
        transition(job.pk, S.PROCESSING)
        assert set_progress(job.pk, 40, "a") and not set_progress(job.pk, 20, "b")
        job.refresh_from_db()
        assert (job.progress_percent, job.stage) == (40, "a")

    def test_full_pipeline_is_declared_but_not_a_bulk_job(self, user):
        assert "FULL_PIPELINE" in AnalyticsJob.JobType.values
        with pytest.raises(services.UnsupportedJobType, match="per video"):
            services.create_job(user, make_asset(user), "FULL_PIPELINE")
        assert AnalyticsJob.objects.count() == 0


# ---------------------------------------------------------------------- task runs
@pytest.mark.django_db
class TestTask:
    def test_vehicle_tracking_pixel_geojson_and_alert(self, user, fake_store, alerts):
        asset = make_asset(user)
        job = make_job(user, asset)
        with mock.patch("apps.videos.vehicles.track_vehicles",
                        side_effect=lambda a, k, url, **kw: completed_tracking(fake_store, a, k)) as tv:
            assert process_drone_video_job(str(job.pk))["status"] == S.COMPLETED
        args = tv.call_args.args
        assert args[0] == str(user.pk) and args[1] == str(asset.id)
        assert args[2].endswith(f"{asset.object_key}?sp=r")
        job.refresh_from_db()
        assert (job.status, job.progress_percent, job.stage) == (S.COMPLETED, 100, "completed")
        geo = job.result_summary["geojson"]
        assert geo["crs"]["properties"]["name"] == "pixel"
        assert geo["features"][0]["geometry"] == {"type": "Point", "coordinates": [640.5, 360.0]}
        video_key = job.result_summary["outputs"]["video"]["key"]
        assert video_key == f"{user.pk}/images/{asset.id}/vehicles/vehicles_tracked.mp4"
        sent = [c.args[1] for c in alerts.call_args_list]
        assert sent == ["job.completed", "alert.detection_triggered"]
        alert = alerts.call_args_list[1].args[2]
        assert alert["count"] == 2 and alert["confidence"] == 0.81 and alert["frame_timestamp"] == 0.5

    def test_geojson_uses_captured_drone_telemetry(self, user, fake_store, alerts):
        make_asset(user, filename="DJI.srt", content_type="application/x-subrip", flight_id="f9",
                   metadata={"telemetry": [{"t": 0.0, "lat": 47.0, "lon": 8.0, "alt": 80.0},
                                           {"t": 2.0, "lat": 47.1, "lon": 8.1, "alt": 81.0}]})
        job = make_job(user, make_asset(user, flight_id="f9"))
        with mock.patch("apps.videos.vehicles.track_vehicles",
                        side_effect=lambda a, k, url, **kw: completed_tracking(fake_store, a, k)):
            process_drone_video_job(str(job.pk))
        geo = AnalyticsJob.objects.get(pk=job.pk).result_summary["geojson"]
        assert "crs" not in geo
        assert geo["features"][0]["geometry"]["coordinates"] == [8.1, 47.1, 81.0]
        assert geo["features"][1]["geometry"]["coordinates"] == [8.0, 47.0, 80.0]
        assert geo["features"][0]["properties"]["position_source"] == "drone_telemetry"

    def test_geospatial_mapping_and_extract_frames(self, user, fake_store, alerts):
        asset = make_asset(user)
        ortho = make_job(user, asset, "GEOSPATIAL_MAPPING")
        with mock.patch("apps.videos.orthomosaic_jobs.run_job",
                        return_value={"state": "completed", "blob": "o.jpg", "output": {"w": 10}}) as rj:
            process_drone_video_job(str(ortho.pk))
        assert rj.call_args.args[:2] == (str(user.pk), str(asset.id))
        ortho.refresh_from_db()
        assert ortho.status == S.COMPLETED and ortho.result_summary["outputs"]["orthomosaic"]["key"] == "o.jpg"

        frames = make_job(user, asset, "EXTRACT_FRAMES", stride=5)
        out = [{"frame_number": 0, "blob_name": "k/images/frame0.jpg", "t": 0.0},
               {"frame_number": 1, "blob_name": "k/images/frame1.jpg", "t": 5.0}]
        with mock.patch("apps.videos.frame_extract.extract_frames", return_value=("stride", out)) as fx:
            process_drone_video_job(str(frames.pk))
        assert fx.call_args.kwargs["stride_seconds"] == 5.0 and fx.call_args.kwargs["video_id"] is None
        frames.refresh_from_db()
        assert frames.result_summary["summary"]["count"] == 2
        assert frames.result_summary["outputs"]["frames"]["keys"] == ["k/images/frame0.jpg", "k/images/frame1.jpg"]

    def test_service_failure_marks_failed(self, user, fake_store, alerts):
        job = make_job(user, make_asset(user))
        with mock.patch("apps.videos.vehicles.track_vehicles",
                        return_value={"state": "failed", "reason": "no ultralytics"}):
            assert process_drone_video_job(str(job.pk))["status"] == S.FAILED
        job.refresh_from_db()
        assert job.status == S.FAILED and job.error_message == "no ultralytics"
        assert alerts.call_args.args[1] == "job.failed"

    def test_unexpected_exception_marks_failed(self, user, fake_store, alerts):
        job = make_job(user, make_asset(user), "EXTRACT_FRAMES")
        with mock.patch("apps.videos.frame_extract.extract_frames", side_effect=RuntimeError("boom")):
            process_drone_video_job(str(job.pk))
        assert AnalyticsJob.objects.get(pk=job.pk).error_message == "boom"

    def test_cancel_during_a_stage_stops_before_the_next(self, user, fake_store, alerts):
        job = make_job(user, make_asset(user))

        def cancel_mid_run(account_id, key, url, **kw):
            assert services.cancel_job(AnalyticsJob.objects.get(pk=job.pk))
            return completed_tracking(fake_store, account_id, key)

        with mock.patch("apps.videos.vehicles.track_vehicles", side_effect=cancel_mid_run), \
                mock.patch.object(runners, "detections_geojson") as geo:
            assert process_drone_video_job(str(job.pk))["status"] == S.CANCELLED
        geo.assert_not_called()                      # the stage after the cancel never ran
        job.refresh_from_db()
        assert job.status == S.CANCELLED and job.result_summary == {}
        assert "job.completed" not in [c.args[1] for c in alerts.call_args_list]

    def test_cancelled_while_queued_is_skipped(self, user, fake_store, alerts):
        job = make_job(user, make_asset(user))
        assert services.cancel_job(job)
        with mock.patch("apps.videos.vehicles.track_vehicles") as tv:
            assert process_drone_video_job(str(job.pk))["skipped"]
        tv.assert_not_called()

    def test_busy_slot_requeues_then_completes(self, user, fake_store, alerts):
        job = services.create_job(user, make_asset(user), "VEHICLE_TRACKING", dispatch_now=False)
        results = iter([{"state": vehicles.BUSY, "reason": "busy"}, None])

        def track(account_id, key, url, **kw):
            result = next(results)
            return result or completed_tracking(fake_store, account_id, key)

        with mock.patch("apps.videos.vehicles.track_vehicles", side_effect=track):
            services.dispatch(job)                   # eager: the re-queue runs immediately
        job.refresh_from_db()
        assert (job.status, job.slot_waits, job.attempt) == (S.COMPLETED, 1, 1)

    def test_busy_too_long_fails(self, user, fake_store, alerts):
        job = make_job(user, make_asset(user))
        AnalyticsJob.objects.filter(pk=job.pk).update(slot_waits=10)
        with mock.patch("apps.videos.vehicles.track_vehicles", return_value={"state": vehicles.BUSY}):
            process_drone_video_job(str(job.pk))
        assert AnalyticsJob.objects.get(pk=job.pk).status == S.FAILED


# --------------------------------------------------------------------------- API
@pytest.mark.django_db
class TestJobsAPI:
    def test_create_job(self, client, user, fake_store, alerts):
        asset = make_asset(user)
        with mock.patch("apps.videos.orthomosaic_jobs.run_job",
                        return_value={"state": "completed", "blob": "o.jpg"}):
            res = client.post("/api/v1/jobs/", {"media_id": str(asset.id), "job_type": "GEOSPATIAL_MAPPING",
                                                "model_version": "3.5", "parameters": {"fast": True}},
                              format="json")
        assert res.status_code == 202, res.content
        body = res.json()
        assert body["job_type"] == "GEOSPATIAL_MAPPING" and body["model_name"] == "opendronemap"
        assert body["parameters"] == {"fast": True} and body["status"] == S.COMPLETED

    @pytest.mark.parametrize("job_type,message", [("FULL_PIPELINE", "per video"),
                                                  ("ANOMALY_SCAN", "unknown job_type")])
    def test_rejected_job_types(self, client, user, fake_store, job_type, message):
        res = client.post("/api/v1/jobs/", {"media_id": str(make_asset(user).id), "job_type": job_type},
                          format="json")
        assert res.status_code == 400 and message in json.dumps(res.json())

    def test_media_must_be_owned_and_uploaded(self, client, user, fake_store):
        other = User.objects.create_user(email="x@example.com", password="pw-123456")
        res = client.post("/api/v1/jobs/", {"media_id": str(make_asset(other).id),
                                            "job_type": "EXTRACT_FRAMES"}, format="json")
        assert res.status_code == 404
        pending = make_asset(user, upload_status=MediaAsset.UploadStatus.PENDING)
        res = client.post("/api/v1/jobs/", {"media_id": str(pending.id), "job_type": "EXTRACT_FRAMES"},
                          format="json")
        assert res.status_code == 409

    def test_list_filters_and_isolation(self, client, user, fake_store):
        asset = make_asset(user)
        a = make_job(user, asset, "VEHICLE_TRACKING")
        b = make_job(user, asset, "EXTRACT_FRAMES")
        transition(b.pk, S.PROCESSING)
        other = User.objects.create_user(email="y@example.com", password="pw-123456")
        make_job(other, make_asset(other))
        assert client.get("/api/v1/jobs/").json()["count"] == 2
        assert [j["id"] for j in client.get("/api/v1/jobs/?status=processing").json()["results"]] == [str(b.id)]
        assert [j["id"] for j in client.get("/api/v1/jobs/?job_type=VEHICLE_TRACKING").json()["results"]] == [str(a.id)]
        AnalyticsJob.objects.filter(pk=a.pk).update(updated_at=timezone.now() - datetime.timedelta(days=2))
        cutoff = (timezone.now() - datetime.timedelta(days=1)).isoformat()
        res = client.get("/api/v1/jobs/", {"updated_after": cutoff})
        assert [j["id"] for j in res.json()["results"]] == [str(b.id)]
        assert client.get(f"/api/v1/jobs/{make_job(other, make_asset(other)).id}/").status_code == 404

    def test_detail_mints_output_urls_when_completed(self, client, user, fake_store):
        job = make_job(user, make_asset(user))
        transition(job.pk, S.PROCESSING)
        transition(job.pk, S.COMPLETED, result_summary={"outputs": {
            "video": {"key": "5/images/x/vehicles/vehicles_tracked.mp4"},
            "frames": {"keys": ["a.jpg", "b.jpg"]}}})
        body = client.get(f"/api/v1/jobs/{job.id}/").json()
        assert body["download_urls"]["video"].endswith("vehicles_tracked.mp4?sp=r")
        assert len(body["download_urls"]["frames"]) == 2

    def test_cancel_endpoint(self, client, user, fake_store):
        job = make_job(user, make_asset(user))
        with mock.patch("apps.notifications.broadcast.push_to_group"):
            res = client.post(f"/api/v1/jobs/{job.id}/cancel/")
            assert res.status_code == 200 and res.json()["status"] == S.CANCELLED
            assert client.post(f"/api/v1/jobs/{job.id}/cancel/").status_code == 409

    def test_retry_endpoint_new_attempt_new_task_id(self, client, user, fake_store, alerts):
        job = make_job(user, make_asset(user), "EXTRACT_FRAMES")
        with mock.patch("apps.videos.frame_extract.extract_frames", side_effect=RuntimeError("vi down")):
            services.dispatch(job)
        job.refresh_from_db()
        first_task = job.celery_task_id
        assert job.status == S.FAILED and job.attempt == 1
        with mock.patch("apps.videos.frame_extract.extract_frames", return_value=("stride", [])):
            res = client.post(f"/api/v1/jobs/{job.id}/retry/")
        assert res.status_code == 202
        body = res.json()
        assert body["status"] == S.COMPLETED and body["attempt"] == 2 and body["error_message"] == ""
        assert body["celery_task_id"] != first_task and body["id"] == str(job.id)
        assert client.post(f"/api/v1/jobs/{job.id}/retry/").status_code == 409
