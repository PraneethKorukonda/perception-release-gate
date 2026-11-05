-- Pedestrian miss rate on a ~1 km grid. Needs GPS (gps.py). Cells with few
-- boxes are left out because a 3-box cell with 2 misses means nothing.
SELECT
    model,
    ROUND(lat, 2)                    AS lat_cell,
    ROUND(lon, 2)                    AS lon_cell,
    COUNT(*)                         AS n_boxes,
    COUNT(DISTINCT video)            AS n_clips,
    ROUND(1 - AVG(det::INT), 4)      AS miss_rate
FROM objects
WHERE cls = 'pedestrian' AND lat IS NOT NULL
GROUP BY ALL
HAVING COUNT(*) >= 50
ORDER BY model, miss_rate DESC;
