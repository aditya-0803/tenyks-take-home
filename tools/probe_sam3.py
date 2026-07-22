#!/usr/bin/env python3
"""Probe the SAM3 result schema of the installed ultralytics version.

Runs the semantic video predictor on the first N sampled frames and prints
the structure of the results (boxes, ids, masks, shapes). Use this when
sam3.py raises a schema error — the output tells us how to fix the parser.

    python tools/probe_sam3.py --video clip.mp4 [--model sam3.pt] [--n 24]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="sam3.pt")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--imgsz", type=int, default=768)
    args = parser.parse_args()

    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames from {args.video}")

    # The SAM3 video predictor requires a real video source (video-mode
    # loader assertion) — write the probe frames to a small temp mp4.
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="sam3_probe_")) / "probe.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    predictor = SAM3VideoSemanticPredictor(
        overrides=dict(
            task="segment", mode="predict", model=args.model,
            imgsz=args.imgsz, conf=0.25, quantize=16, save=False, verbose=False,
        )
    )
    results = predictor(source=str(tmp), text=["person"], stream=True)

    for i, r in enumerate(results):
        print(f"\n--- frame {i} ---")
        print("result type:", type(r).__name__)
        print("attrs:", [a for a in dir(r) if not a.startswith("_")][:30])
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            print("boxes:", len(boxes), "| id:", getattr(boxes, "id", None))
            if len(boxes):
                print("xyxy[0]:", boxes.xyxy[0].tolist(),
                      "conf[0]:", float(boxes.conf[0]) if boxes.conf is not None else None)
        masks = getattr(r, "masks", None)
        if masks is not None:
            print("masks.data shape:", tuple(masks.data.shape))
        if i >= 3:
            print("\n(stopping after 4 frames — schema should be clear)")
            break


if __name__ == "__main__":
    main()
