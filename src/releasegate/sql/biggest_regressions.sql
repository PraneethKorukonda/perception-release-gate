-- Where did the candidate lose the most recall, compared box for box?
-- Same boxes for both models, so the comparison is paired. Exploratory: with this
-- many slices, a few will look bad by chance. Use it to decide what to look at,
-- then confirm with a gate check.
WITH paired AS (
    SELECT
        c.cls, c.timeofday, c.weather, c."size",
        c.video,
        c.det::INT AS det_current,
        n.det::INT AS det_candidate
    FROM objects c
    JOIN objects n USING (gt_id)
    WHERE c.model = 'current' AND n.model = 'candidate'
)
SELECT
    cls, timeofday, weather, "size",
    COUNT(*)                                            AS n_boxes,
    COUNT(DISTINCT video)                               AS n_clips,
    ROUND(AVG(det_current), 4)                          AS recall_current,
    ROUND(AVG(det_candidate), 4)                        AS recall_candidate,
    ROUND(AVG(det_candidate) - AVG(det_current), 4)     AS diff
FROM paired
GROUP BY ALL
HAVING COUNT(*) >= 200 AND COUNT(DISTINCT video) >= 5
ORDER BY diff
LIMIT 20;
