# Kiosk Analytics — People Count & Dwell Time from Video

Counts the distinct people who queue for / use a self-service kiosk in
CCTV footage and measures each person's dwell time (queue + kiosk use),
rendering an annotated overlay video and evaluating itself against
hand-labelled ground truth.

## Pipeline

```
video ──> detect (YOLO11 / RT-DETR) ──> track (ByteTrack / BoT-SORT / OC-SORT)
      ──> offline re-ID stitching (OSNet + Hungarian) ──> zone analytics
      ──> persons.csv + summary.json + overlay.mp4 [+ metrics.json]
```

Design choices worth knowing:

- **Tracklets, not tracks.** The online tracker is configured to prefer
  breaking a track over swapping identities; an offline stitching pass
  re-links fragments using appearance embeddings solved as a global
  assignment problem. This is what makes the distinct-person count robust
  to occlusion and leave-and-return behaviour.
- **Explicit engagement definition.** A person is engaged iff their feet
  are inside the kiosk zone polygon for at least `min_engagement_s` total.
  Dwell = sum of in-zone segments (policy documented in
  `docs/LABELING.md`, mirrored by the ground-truth protocol).
- **fps paranoia.** CCTV containers lie about frame rate (this footage
  says 100 fps nominal, is ~30 fps actual); timestamps come from
  ffprobe's average frame rate.

## Setup

```bash
git clone <repo> && cd kiosk-analytics
pip install -r requirements.txt
```

On Google Colab:

```python
!pip install -q -r requirements.txt
```

Model weights (YOLO, OSNet) download automatically on first run.
Everything fits in < 4 GB VRAM at default settings (16 GB budget).

## Run

```bash
python run.py --video path/to/video.mp4 --config config/default.yaml --out runs/exp1
```

With evaluation against ground truth:

```bash
python run.py --video clip.mp4 --gt labels/gt.csv --out runs/exp1
```

Outputs in `--out`:

| File | Contents |
|---|---|
| `persons.csv` | per-person dwell, segments, engaged flag, source tracklets |
| `summary.json` | distinct-person count, dwell stats, runtime |
| `overlay.mp4` | boxes, persistent IDs, running dwell, live counts, zone |
| `metrics.json` | count accuracy, dwell MAE/MAPE vs GT (if `--gt`) |

## Configure

All knobs live in `config/default.yaml` (detector/tracker choice, zone
polygon, engagement thresholds, stitching gates). To adapt to a new
camera view, redraw the zone:

```bash
python tools/draw_zone.py --video clip.mp4 --t 30      # interactive (local)
python tools/extract_frame.py --video clip.mp4 --grid  # headless fallback
```

then paste the printed `zone_polygon` into the config.

## Ground truth

Label the video per `docs/LABELING.md` into a CSV
(`person_id,start_s,end_s`, one row per in-zone segment).

## Tests

```bash
pip install pytest && pytest
```

Covers the pure-logic core (zone tests, dwell segmentation/hysteresis,
stitch gating + assignment, evaluation matching) without needing a GPU.

## Repo layout

```
run.py                     CLI entrypoint
config/default.yaml        pipeline configuration
src/kiosk_analytics/
  video.py                 fps-safe frame sampling
  detect.py                ultralytics detector wrapper
  track.py                 boxmot tracker adapter
  tracklets.py             tracklet store + best-crop collection
  stitch.py                offline re-ID stitching (global assignment)
  zone.py                  polygon membership
  analytics.py             dwell segments, engagement rule, count
  viz.py                   overlay renderer
  eval.py                  metrics vs ground truth
  pipeline.py              orchestration
tools/                     zone drawing / frame extraction helpers
docs/LABELING.md           ground-truth labelling protocol
tests/                     unit tests for the pure-logic core
```
