-- DVSA on Databricks — reporting queries over inference_results.
-- Pair with schemas.sql. These power the "Reporting" stage of the pipeline.

-- Detections per model (overall throughput / coverage).
SELECT model_name,
       COUNT(*)                AS frames_scored,
       SUM(num_detections)     AS total_detections,
       AVG(num_detections)     AS avg_detections_per_frame,
       SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors
FROM drone.video.inference_results
GROUP BY model_name
ORDER BY total_detections DESC;

-- Per-video detection timeline (for dashboards / anomaly spotting).
SELECT video_id,
       frame_index,
       num_detections
FROM drone.video.inference_results
WHERE error IS NULL
ORDER BY video_id, frame_index;

-- Frames with the most detections (hotspots worth reviewing).
SELECT video_id, frame_index, num_detections, model_name
FROM drone.video.inference_results
WHERE error IS NULL
ORDER BY num_detections DESC
LIMIT 20;

-- Extract a field from the JSON payload (example: a summary verdict).
SELECT video_id,
       frame_index,
       get_json_object(result_json, '$.summary.count') AS summary_count
FROM drone.video.inference_results
WHERE error IS NULL;
