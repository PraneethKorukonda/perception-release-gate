-- Pedestrian recall by city (Waymo: phx / sf / other) and time of day.
-- A gap between cities can be the detector or just a different mix of distances,
-- so check median_distance_m before reading it as a location effect.
SELECT
    model,
    location,
    timeofday,
    COUNT(*)                                  AS n_boxes,
    COUNT(DISTINCT video)                     AS n_clips,
    ROUND(MEDIAN(distance_m), 1)              AS median_distance_m,
    ROUND(AVG(det::INT), 4)                   AS recall,
    ROUND(AVG(CASE WHEN near THEN det::INT END), 4) AS recall_within_40m
FROM objects
WHERE cls = 'pedestrian'
GROUP BY ALL
ORDER BY location, timeofday, model;
