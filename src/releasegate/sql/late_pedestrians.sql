-- In-path pedestrians the candidate detected too late to stop for, but the
-- current model didn't. These are the clips to watch before signing off.
SELECT
    c.video,
    c.track_id,
    c.first_frame,
    c.timeofday,
    c.weather,
    ROUND(c.distance_at_appear_m, 1)   AS appeared_at_m,
    ROUND(c.distance_at_det_m, 1)      AS current_detected_at_m,
    ROUND(n.distance_at_det_m, 1)      AS candidate_detected_at_m,
    ROUND(n.speed_at_det_mps, 1)       AS speed_mps,
    ROUND(n.stop_margin_m, 1)          AS candidate_margin_m
FROM tracks c
JOIN tracks n USING (track_id)
WHERE c.model = 'current' AND n.model = 'candidate'
  AND c.cls = 'pedestrian'
  AND c.late_in_path = 0 AND n.late_in_path = 1
ORDER BY candidate_margin_m NULLS FIRST;
