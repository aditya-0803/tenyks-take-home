#!/usr/bin/env python3
"""Interactively draw the kiosk zone polygon on a video frame (local only;
needs a display). Left-click to add vertices, 'u' to undo, 's' to print
the YAML snippet and save zone.png, 'q' to quit.

    python tools/draw_zone.py --video clip.mp4 --t 30
"""

from __future__ import annotations

import argparse

import cv2

points: list[list[int]] = []


def on_mouse(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--t", type=float, default=0.0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.t * 1000)
    ok, base = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read frame")

    cv2.namedWindow("zone")
    cv2.setMouseCallback("zone", on_mouse)
    while True:
        frame = base.copy()
        for i, p in enumerate(points):
            cv2.circle(frame, tuple(p), 4, (0, 0, 255), -1)
            if i:
                cv2.line(frame, tuple(points[i - 1]), tuple(p), (0, 255, 0), 2)
        if len(points) > 2:
            cv2.line(frame, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 1)
        cv2.imshow("zone", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u") and points:
            points.pop()
        elif key == ord("s"):
            print("zone_polygon:")
            for x, y in points:
                print(f"  - [{x}, {y}]")
            cv2.imwrite("zone.png", frame)
            print("Saved zone.png")
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
