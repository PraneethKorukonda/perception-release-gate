-- Clips ranked by pedestrian boxes the current model found and the candidate missed.
-- Good starting list for a side-by-side video review.
SELECT
    c.video,
    ANY_VALUE(c.timeofday)                                   AS timeofday,
    ANY_VALUE(c.weather)                                     AS weather,
    ANY_VALUE(c.scene)                                       AS scene,
    COUNT(*)                                                 AS ped_boxes,
    SUM(CASE WHEN c.det AND NOT n.det THEN 1 ELSE 0 END)     AS lost_by_candidate,
    SUM(CASE WHEN n.det AND NOT c.det THEN 1 ELSE 0 END)     AS gained_by_candidate
FROM objects c
JOIN objects n USING (gt_id)
WHERE c.model = 'current' AND n.model = 'candidate' AND c.cls = 'pedestrian'
GROUP BY c.video
ORDER BY lost_by_candidate - gained_by_candidate DESC
LIMIT 25;
