-- DVSA on Databricks — Delta table schemas for the ETL -> Feature -> Inference
-- -> Reporting pipeline. Mirrors the youtube-video-intelligence layout:
-- raw_frames -> tracks -> features -> inference_results.
--
-- Run in a SQL notebook/warehouse. Uses Unity Catalog three-level names;
-- replace `drone` with your catalog.schema as needed.

CREATE CATALOG IF NOT EXISTS drone;
CREATE SCHEMA IF NOT EXISTS drone.video;

-- 1) ETL: raw frame metadata (one row per sampled frame).
CREATE TABLE IF NOT EXISTS drone.video.raw_frames (
  video_id      BIGINT,
  frame_index   BIGINT,
  timestamp     DOUBLE,
  width         INT,
  height        INT,
  thumbnail_b64 STRING,
  sensor_meta   STRING,          -- JSON: gps/altitude/camera
  ingested_at   TIMESTAMP
) USING DELTA;

-- 2) Object tracks (one row per frame; tracks stored as a JSON array).
CREATE TABLE IF NOT EXISTS drone.video.tracks (
  video_id             BIGINT,
  frame_index          BIGINT,
  tracks               STRING,   -- JSON array of {id,bbox,velocity,timestamp,label}
  sensor_meta          STRING,
  query                STRING
) USING DELTA;

-- 3) Precomputed features (numeric aggregates used by inference).
CREATE TABLE IF NOT EXISTS drone.video.features (
  video_id             BIGINT,
  frame_index          BIGINT,
  precomputed_features STRING    -- JSON map of numeric features
) USING DELTA;

-- 4) Inference results (written by run_inference_batch / stream_inference).
CREATE TABLE IF NOT EXISTS drone.video.inference_results (
  video_id       BIGINT,
  frame_index    BIGINT,
  model_name     STRING,
  result_json    STRING,         -- full DVSA payload
  num_detections INT,
  error          STRING
) USING DELTA;
