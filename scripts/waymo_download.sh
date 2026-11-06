#!/usr/bin/env bash
# Download a Waymo Open Dataset (perception v2.0.1) eval set, without the full
# few-hundred-GB validation split.
#
# Before running, once:
#   1. Accept the license at waymo.com/open (Download page) with your Google account.
#   2. Install the Google Cloud CLI:   brew install --cask google-cloud-sdk
#   3. Log in with that same account:  gcloud auth login
#
# Steps:
#   1. Every validation segment's stats file (~25 KB each) -> pick N segments,
#      all night/dawn/dusk/rain ones with pedestrians plus a random fill.
#   2. The five components the project uses, for those segments only.
#      camera_image is ~350 MB per segment, so N=40 is about 14 GB.
set -euo pipefail

N=${N:-40}
DEST=${DEST:-waymo/validation}
BUCKET=gs://waymo_open_dataset_v_2_0_1/validation

mkdir -p "$DEST/stats"
echo "1/2  stats for every validation segment"
gcloud storage cp --no-clobber "$BUCKET/stats/*.parquet" "$DEST/stats/"

releasegate waymo-pick --stats "$DEST/stats" --n "$N" --out "$DEST/segments.csv"

echo "2/2  components for the picked segments"
# Uses `gcloud storage`, not `gsutil -m cp -I`: on macOS the gsutil version read
# only part of the list (Python multiprocessing trouble) and still exited cleanly.
for C in camera_box camera_calibration vehicle_pose camera_image; do
  mkdir -p "$DEST/$C"
  tail -n +2 "$DEST/segments.csv" | cut -d, -f1 | sed "s|^|$BUCKET/$C/|; s|$|.parquet|" \
    | gcloud storage cp --no-clobber --read-paths-from-stdin "$DEST/$C/"
  echo "  $C: $(ls "$DEST/$C" | wc -l | tr -d ' ') files"
done
echo "done -> $DEST"
