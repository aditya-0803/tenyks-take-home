#!/usr/bin/env python3
"""Extract a reference frame (optionally with a coordinate grid) for zone drawing.

    python tools/extract_frame.py --video clip.mp4 --t 30 --grid --out frame.png
"""

from __future__ import annotations

import argparse

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--t", type=float, default=0.0, help="Timestamp (s)")
    parser.add_argument("--out", default="frame.png")
    parser.add_argument("--grid", action="store_true", help="Overlay a 100px coordinate grid")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame at t={args.t}s")

    if args.grid:
        h, w = frame.shape[:2]
        for x in range(0, w, 100):
            cv2.line(frame, (x, 0), (x, h), (0, 255, 255), 1)
            cv2.putText(frame, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for y in range(0, h, 100):
            cv2.line(frame, (0, y), (w, y), (0, 255, 255), 1)
            cv2.putText(frame, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    cv2.imwrite(args.out, frame)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
