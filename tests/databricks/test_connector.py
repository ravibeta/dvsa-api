"""Offline unit tests for the DVSA Databricks connector.

These run without a live Databricks workspace or DVSA endpoint:

* pure mapping/batching tests use plain dicts and a stub client,
* remote-client tests mock HTTP with ``requests_mock``,
* Spark-backed tests use a local ``pyspark`` session and are skipped when
  ``pyspark`` is not installed (so the suite still runs on a minimal env).
"""

from __future__ import annotations

import pytest

from dvsa_databricks_connector import (
    DVSAClient,
    DVSAConfig,
    InClusterDVSAClient,
    build_client,
    config_from_env,
    result_to_row,
    row_to_context,
    run_inference_on_rows,
)
from dvsa_databricks_connector.connector import _count_detections, iter_batches


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _StubClient(DVSAClient):
    """Echoes a fixed detection payload; counts calls."""

    def __init__(self):
        self.calls = 0

    def infer_one(self, context, model_name):
        self.calls += 1
        return {"detections": [{"label": "car"}, {"label": "van"}],
                "summary": {"count": 2}, "model_name": model_name}


class _FlakyClient(DVSAClient):
    """Raises on the ``fail_on``-th call (1-indexed); succeeds otherwise.

    Keys off call order rather than a context field because ``row_to_context``
    intentionally drops columns DVSA does not understand.
    """

    def __init__(self, fail_on):
        self.fail_on = fail_on
        self.calls = 0

    def infer_one(self, context, model_name):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("boom")
        return {"detections": [], "summary": {"count": 0}}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class TestConfig:
    def test_from_env(self):
        cfg = config_from_env({"DVSA_ENDPOINT": "https://x/api/v1",
                               "DVSA_API_KEY": "k", "DVSA_MODE": "remote"})
        assert cfg.endpoint == "https://x/api/v1"
        assert cfg.infer_url() == "https://x/api/v1/analytics/infer"
        cfg.validate()

    def test_validate_requires_endpoint(self):
        with pytest.raises(ValueError):
            DVSAConfig(mode="remote", api_key="k").validate()

    def test_validate_requires_api_key(self):
        with pytest.raises(ValueError):
            DVSAConfig(mode="remote", endpoint="https://x").validate()

    def test_in_cluster_needs_neither(self):
        DVSAConfig(mode="in_cluster").validate()  # no raise

    def test_redacted_masks_key(self):
        cfg = DVSAConfig(mode="remote", endpoint="https://x", api_key="secret")
        assert cfg.redacted()["api_key"] == "***"
        assert "secret" not in str(cfg.redacted())

    def test_bad_mode(self):
        with pytest.raises(ValueError):
            DVSAConfig(mode="nope")


# --------------------------------------------------------------------------- #
# Pure mapping
# --------------------------------------------------------------------------- #
class TestMapping:
    def test_row_to_context_json_strings(self):
        row = {
            "video_id": 1, "frame_index": 2,
            "tracks": '[{"id": 1, "bbox": [1,2,3,4], "velocity": [0.1,0.2], "timestamp": 0.5}]',
            "sensor_meta": {"gps": [1.0, 2.0], "altitude": 100},
            "precomputed_features": '{"mean_speed": 0.5}',
            "query": "count cars",
        }
        ctx = row_to_context(row)
        assert ctx["tracks"][0]["id"] == 1
        assert ctx["sensor_meta"]["altitude"] == 100
        assert ctx["precomputed_features"]["mean_speed"] == 0.5
        assert ctx["query"] == "count cars"
        assert "frames" not in ctx  # none supplied

    def test_include_frames_toggle(self):
        row = {"frames": [{"frame_index": 0}], "tracks": []}
        assert "frames" in row_to_context(row, include_frames=True)
        assert "frames" not in row_to_context(row, include_frames=False)

    def test_empty_row_is_empty_context(self):
        assert row_to_context({}) == {}

    def test_single_dict_track_wrapped_in_list(self):
        ctx = row_to_context({"tracks": {"id": 9}})
        assert ctx["tracks"] == [{"id": 9}]

    def test_malformed_json_string_passthrough(self):
        # A non-JSON query string must survive intact, not blow up.
        ctx = row_to_context({"query": "not-json {["})
        assert ctx["query"] == "not-json {["


class TestResultRow:
    def test_identity_passthrough_and_json(self):
        row = {"video_id": 5, "frame_index": 7}
        out = result_to_row(row, {"detections": [1, 2, 3]}, "m")
        assert out["video_id"] == 5 and out["frame_index"] == 7
        assert out["model_name"] == "m"
        assert out["num_detections"] == 3
        assert '"detections"' in out["result_json"]
        assert out["error"] is None

    def test_error_row(self):
        out = result_to_row({"video_id": 1}, {}, "m", error="timeout")
        assert out["error"] == "timeout"
        assert out["num_detections"] == 0

    @pytest.mark.parametrize("result,expected", [
        ({"detections": [1, 2]}, 2),
        ({"tracks": [1]}, 1),
        ({"results": []}, 0),
        ({"summary": {"count": 4}}, 4),
        ({}, 0),
        ("not-a-dict", 0),
    ])
    def test_count_detections(self, result, expected):
        assert _count_detections(result) == expected


class TestBatching:
    def test_iter_batches(self):
        assert list(iter_batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    def test_iter_batches_bad_size(self):
        with pytest.raises(ValueError):
            list(iter_batches([1], 0))

    def test_run_inference_on_rows_metrics(self):
        rows = [{"video_id": i, "tracks": []} for i in range(5)]
        client = _StubClient()
        out = run_inference_on_rows(rows, "m", client, batch_size=2)
        assert client.calls == 5
        assert out["metrics"]["rows"] == 5.0
        assert out["metrics"]["batches"] == 3.0
        assert out["metrics"]["detections"] == 10.0  # 2 per row
        assert out["metrics"]["errors"] == 0.0

    def test_run_inference_isolates_errors(self):
        rows = [{"video_id": i, "tracks": []} for i in range(3)]
        out = run_inference_on_rows(rows, "m", _FlakyClient(fail_on=2), batch_size=10)
        assert out["metrics"]["errors"] == 1.0
        errored = [r for r in out["results"] if r["error"]]
        assert len(errored) == 1 and "boom" in errored[0]["error"]
        # The other two rows still produced results.
        assert len(out["results"]) == 3


# --------------------------------------------------------------------------- #
# Remote client (mocked HTTP)
# --------------------------------------------------------------------------- #
class TestRemoteClient:
    def _cfg(self, **kw):
        base = dict(mode="remote", endpoint="https://dvsa.test/api/v1",
                    api_key="k", backoff_factor=0.0)
        base.update(kw)
        return DVSAConfig(**base)

    def test_infer_posts_context_and_returns_json(self, requests_mock):
        cfg = self._cfg()
        requests_mock.post(cfg.infer_url(), json={"summary": {"count": 1}})
        client = build_client(cfg)
        out = client.infer({"tracks": []}, "m")
        assert out == {"summary": {"count": 1}}
        # Auth header + payload shape.
        req = requests_mock.request_history[0]
        assert req.headers["Authorization"] == "Bearer k"
        assert req.json() == {"model_name": "m", "context": {"tracks": []}}

    def test_retries_then_succeeds(self, requests_mock):
        cfg = self._cfg(max_retries=2)
        requests_mock.post(cfg.infer_url(), [
            {"status_code": 503},
            {"status_code": 200, "json": {"ok": True}},
        ])
        out = build_client(cfg).infer({}, "m")
        assert out == {"ok": True}
        assert requests_mock.call_count == 2

    def test_exhausts_retries_raises(self, requests_mock):
        cfg = self._cfg(max_retries=1)
        requests_mock.post(cfg.infer_url(), status_code=500)
        with pytest.raises(RuntimeError):
            build_client(cfg).infer({}, "m")
        assert requests_mock.call_count == 2  # initial + 1 retry


# --------------------------------------------------------------------------- #
# In-cluster client
# --------------------------------------------------------------------------- #
class TestInClusterClient:
    def test_calls_infer_fn(self):
        client = InClusterDVSAClient(infer_fn=lambda ctx, m: {"m": m, "n": len(ctx)})
        assert client.infer({"a": 1}, "x") == {"m": "x", "n": 1}

    def test_missing_fn_raises(self):
        with pytest.raises(RuntimeError):
            InClusterDVSAClient().infer({}, "x")

    def test_non_dict_return_raises(self):
        with pytest.raises(TypeError):
            InClusterDVSAClient(infer_fn=lambda c, m: [1, 2]).infer({}, "x")


# --------------------------------------------------------------------------- #
# Spark-backed end-to-end (skipped without pyspark)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    sess = (
        SparkSession.builder.appName("dvsa-tests")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield sess
    sess.stop()


class TestSparkFlow:
    def test_run_inference_batch_maps_and_returns_results(self, spark):
        from dvsa_databricks_connector import run_inference_batch

        rows = [
            {"video_id": 1, "frame_index": 0,
             "tracks": '[{"id": 1, "bbox": [1,2,3,4]}]',
             "sensor_meta": '{"altitude": 100}', "query": "count"},
            {"video_id": 1, "frame_index": 5,
             "tracks": "[]", "sensor_meta": '{"altitude": 101}', "query": "count"},
        ]
        df = spark.createDataFrame(rows)
        result_df = run_inference_batch(
            df, model_name="m", client=_StubClient(),
            batch_size=1, mlflow_run=False,
        )
        collected = {r["frame_index"]: r for r in result_df.collect()}
        assert set(collected) == {0, 5}
        assert collected[0]["model_name"] == "m"
        assert collected[0]["num_detections"] == 2
        assert collected[0]["error"] is None
        # result_json is a valid JSON string carrying the DVSA payload.
        import json
        assert json.loads(collected[0]["result_json"])["summary"]["count"] == 2

    def test_result_schema_is_stable(self, spark):
        from dvsa_databricks_connector.connector import _results_to_spark_df, RESULT_COLUMNS

        df = _results_to_spark_df(spark, [
            {"video_id": "1", "frame_index": None, "model_name": "m",
             "result_json": "{}", "num_detections": 0, "error": None},
        ])
        assert tuple(df.columns) == RESULT_COLUMNS
        row = df.collect()[0]
        assert row["video_id"] == 1  # coerced from str
        assert row["frame_index"] is None
