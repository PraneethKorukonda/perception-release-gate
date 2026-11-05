#!/usr/bin/env bash
# Full run on the BDD100K tracking val split, on a laptop.
# Set BDD to wherever you unzipped BDD100K. On an M1/M2 Mac the two inference
# steps take 30-45 minutes each at imgsz 1280; add --limit 40 to try it first.
set -euo pipefail

BDD=${BDD:-$HOME/data/bdd100k}
DEVICE=${DEVICE:-mps}
CURRENT=${CURRENT:-yolo11s.pt}
CANDIDATE=${CANDIDATE:-yolo11m.pt}

# GPS/speed files come in a separate "info" download. Point INFO at the folder
# of per-video .json files; clips without one fall back to a typical speed.
INFO=${INFO:-$BDD/info/100k/val}
EXTRA=()
if [ -d "$INFO" ]; then EXTRA+=(--info "$INFO"); fi

releasegate ingest --labels "$BDD/labels/box_track_20/val" \
                   --estimate-lighting "$BDD/images/track/val" \
                   --out data/val ${EXTRA[@]+"${EXTRA[@]}"}

for W in "$CURRENT" "$CANDIDATE"; do
  NAME=$(basename "$W" .pt)
  releasegate infer --data data/val --images "$BDD/images/track/val" \
                    --weights "$W" --out "runs/$NAME" --device "$DEVICE"
  releasegate match --data data/val --run "runs/$NAME"
done

releasegate scenario --data data/val --name night_pedestrians \
    --where "cls == 'pedestrian' and night" --out datasets/night_pedestrians.json

releasegate gate --data data/val \
    --current "runs/$(basename "$CURRENT" .pt)" \
    --candidate "runs/$(basename "$CANDIDATE" .pt)" \
    --set "datasets/*.json" \
    --out reports/val
