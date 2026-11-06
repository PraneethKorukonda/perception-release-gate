#!/usr/bin/env bash
# Full run on a downloaded Waymo eval set (scripts/waymo_download.sh).
# On a MacBook Air M2 (8 GB), 1280 px with yolo11s ran out of memory and slowed
# to a crawl, so the defaults are 960 px and conf 0.05. yolo11n ran at ~12
# frames/s at 1280. Use the same IMGSZ and CONF for both models, or you're
# comparing input settings, not models.
set -euo pipefail

ROOT=${ROOT:-waymo/validation}
DEVICE=${DEVICE:-mps}
CANDIDATE=${CANDIDATE:-yolo11s.pt}
CURRENT=${CURRENT:-yolo11n.pt}
IMGSZ=${IMGSZ:-960}
CONF=${CONF:-0.05}

releasegate waymo-ingest --root "$ROOT" --out data/waymo --images data/waymo/images

for W in "$CURRENT" "$CANDIDATE"; do
  NAME=$(basename "$W" .pt)
  releasegate infer --data data/waymo --images data/waymo/images --classes waymo \
                    --weights "$W" --out "runs/$NAME" --device "$DEVICE" \
                    --imgsz "$IMGSZ" --conf "$CONF"
  releasegate match --data data/waymo --run "runs/$NAME"
done

releasegate gate --data data/waymo --config waymo \
    --current "runs/$(basename "$CURRENT" .pt)" \
    --candidate "runs/$(basename "$CANDIDATE" .pt)" \
    --out reports/waymo
