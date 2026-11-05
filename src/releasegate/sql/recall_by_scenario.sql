-- Frame-level recall for each model, by class, time of day and weather.
-- Small cells are noisy: look at n_boxes and n_clips before reading anything into a gap.
SELECT
    model,
    cls,
    timeofday,
    weather,
    COUNT(*)                         AS n_boxes,
    COUNT(DISTINCT video)            AS n_clips,
    ROUND(AVG(det::INT), 4)          AS recall
FROM objects
GROUP BY ALL
ORDER BY cls, timeofday, weather, model;
