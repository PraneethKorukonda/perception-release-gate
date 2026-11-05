-- Objects the candidate keeps losing and re-finding. A downstream tracker sees
-- each dropout as a coast or a new track, so these are worth a look even when
-- recall is fine.
SELECT
    n.video,
    n.track_id,
    n.cls,
    n.n_frames,
    n.n_det_frames,
    c.dropouts AS dropouts_current,
    n.dropouts AS dropouts_candidate
FROM tracks c
JOIN tracks n USING (track_id)
WHERE c.model = 'current' AND n.model = 'candidate'
ORDER BY n.dropouts - c.dropouts DESC
LIMIT 25;
