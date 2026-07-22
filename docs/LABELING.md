# Ground-Truth Labelling Protocol

The evaluation is only meaningful if the ground truth and the system share
one definition of "engagement". This protocol mirrors the pipeline's dwell
policy exactly.

## Definitions

- **Kiosk zone**: the polygon in `config/default.yaml` (customer side of
  the kiosks plus the queue corridor). Overlay it on a frame
  (`tools/extract_frame.py` + the zone preview in the overlay video) and
  keep it visible while labelling.
- **In-zone**: the person's feet (bottom-centre of their body) are inside
  the polygon.
- **Passby (per visit)**: an in-zone visit shorter than `min_segment_s`
  (default 12 s) is a walk-through — do NOT label that segment, even for
  a person who has other, longer visits. (Brief occlusion gaps <= 3 s
  within one visit do not split it.)
- **Dwell time (policy a)**: the sum of a person's in-zone segments that
  each last >= 12 s. Time outside the zone between segments is excluded.
- **Engaged person**: has at least one labelled (>= 12 s) segment.
  People with only passby visits are NOT labelled.

## Procedure

1. Watch the clip at 1x (slower where crowded). For every person who
   enters the zone and plausibly queues/uses a kiosk, assign a label
   (`P1`, `P2`, ... in order of first appearance).
2. Record one row per in-zone segment in `gt.csv`:

   ```csv
   person_id,start_s,end_s
   P1,12.4,105.0
   P2,31.0,58.5
   P2,102.0,140.2
   ```

   - `start_s`: first moment the person's feet are inside the zone
     (video time, seconds).
   - `end_s`: the moment they leave the zone.
   - A person who leaves and returns gets multiple rows (their dwell is
     the sum).
3. Timing tolerance: label to the nearest 0.5 s; boundary judgement calls
   of +/- 2 s are expected and within noise.
4. Edge cases (decisions must be consistent):
   - **Children** accompanying an adult, whether carried or wandering,
     are NOT labelled (they are not kiosk customers). Note each occurrence
     in a comments column/file for the failure-mode analysis.
   - **Staff** assisting at the kiosk are NOT labelled; note occurrences.
   - **Walk-throughs** (in zone < 8 s total) are NOT labelled.
   - **Give-ups** (queued > 8 s then left without using the kiosk) ARE
     labelled: they engaged with the queue.
5. Record clip name, labeller, date, and any ambiguous cases in a short
   notes file next to `gt.csv`.
