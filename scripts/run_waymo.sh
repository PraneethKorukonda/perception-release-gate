#!/usr/bin/env bash
# Full run on a downloaded Waymo eval set (scripts/waymo_download.sh).
# On an M1/M2 Mac, inference at 1280 px is roughly 15-25 frames/s per model,
# so 40 segments (~8,000 front-camera frames) take about 5-10 minutes each.
set -euo pipefail

ROOT=${ROOT:-waymo/validation}
DEVICE=${DEVICE:-mps}
CURRENT=${CURRENT:-yolo11s.pt}
CANDIDATE=${CANDIDATE:-yolo11m.pt}

releasegate waymo-ingest --root "$ROOT" --out data/waymo --images data/waymo/images

for W in "$CURRENT" "$CANDIDATE"; do
  NAME=$(basename "$W" .pt)
  releasegate infer --data data/waymo --images data/waymo/images --classes waymo \
                    --weights "$W" --out "runs/$NAME" --device "$DEVICE"
  releasegate match --data data/waymo --run "runs/$NAME"
done

releasegate gate --data data/waymo --config waymo \
    --current "runs/$(basename "$CURRENT" .pt)" \
    --candidate "runs/$(basename "$CANDIDATE" .pt)" \
    --out reports/waymo
