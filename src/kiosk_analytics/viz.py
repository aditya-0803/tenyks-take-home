"""Annotated overlay video: box + persistent ID + running dwell per person,
live engaged-now / unique-total counters, and the zone polygon.

Rendered as a second pass over the video so that overlays reflect the
final (stitched) identities rather than raw tracker IDs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .analytics import PersonTimeline
from .config import Config
from .video import VideoReader

log = logging.getLogger(__name__)

_PALETTE = [
    (66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (124, 179, 66),
    (92, 107, 192), (240, 98, 146), (0, 137, 123), (253, 216, 53),
]


def _color(pid: int) -> tuple[int, int, int]:
    return _PALETTE[pid % len(_PALETTE)]


def _draw_label(frame, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y = max(y, th + 8)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y), color, -1)
    cv2.putText(frame, text, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def render_overlay(
    video_path: str | Path,
    out_path: str | Path,
    timelines: list[PersonTimeline],
    engaged_pids: set[int],
    min_engagement_s: float,
    cfg: Config,
) -> Path:
    reader = VideoReader(video_path, cfg.video.target_fps)
    out_path = Path(out_path)

    # frame_idx -> list of (pid, box, dwell, in_zone)
    per_frame: dict[int, list] = {}
    engaged_at: dict[int, float] = {}  # pid -> time it crossed the threshold
    for tl in timelines:
        for i, fidx in enumerate(tl.frame_indices):
            per_frame.setdefault(int(fidx), []).append(
                (tl.pid, tl.boxes[i], float(tl.cum_dwell[i]), bool(tl.in_zone[i]))
            )
        if tl.pid in engaged_pids:
            crossed = np.searchsorted(tl.cum_dwell, min_engagement_s)
            idx = min(int(crossed), len(tl.times) - 1)
            engaged_at[tl.pid] = float(tl.times[idx])

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        reader.sample_fps,
        (reader.width, reader.height),
    )
    zone_pts = np.array(cfg.zone_polygon, dtype=np.int32)

    for sample in reader:
        frame = sample.image
        if cfg.viz.show_zone:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone_pts], (60, 160, 60))
            frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
            cv2.polylines(frame, [zone_pts], True, (60, 200, 60), 2)

        entries = per_frame.get(sample.index, [])
        for pid, box, dwell, in_zone in entries:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            # In-zone: person's colour, bold. Out-of-zone: thin gray
            # (tracked but not currently accruing dwell).
            color = _color(pid) if in_zone else (140, 140, 140)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if in_zone else 1)
            _draw_label(frame, f"ID {pid} | {dwell:.0f}s", x1, y1, color)

        now_engaged = sum(1 for _, _, _, in_zone in entries if in_zone)
        unique_total = sum(1 for t0 in engaged_at.values() if t0 <= sample.t)
        panel = [
            f"In zone now: {now_engaged}",
            f"Unique people (total): {unique_total}",
        ]
        cv2.rectangle(frame, (10, 10), (330, 76), (30, 30, 30), -1)
        for i, line in enumerate(panel):
            cv2.putText(frame, line, (20, 38 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)

        writer.write(frame)

    writer.release()
    log.info("Overlay written to %s", out_path)
    return out_path
