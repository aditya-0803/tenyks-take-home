"""Video reading with fps-safe frame sampling.

CCTV files often carry a bogus nominal frame rate (this footage reports
100 fps nominal vs ~30 fps actual), so we probe the *average* frame rate
with ffprobe and fall back to OpenCV only if ffprobe is unavailable.
Dwell times are computed from timestamps derived here, so getting fps
right matters more than anywhere else in the pipeline.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FrameSample:
    index: int        # frame index in the source video
    t: float          # timestamp in seconds
    image: np.ndarray  # BGR frame


def probe_fps(path: str | Path) -> float:
    """Return the average frame rate, preferring ffprobe over OpenCV."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=30, check=True,
        )
        rate = json.loads(out.stdout)["streams"][0]["avg_frame_rate"]
        fps = float(Fraction(rate))
        if fps > 0:
            return fps
    except (Exception,):  # noqa: BLE001 - any failure falls through to OpenCV
        log.warning("ffprobe unavailable or failed; falling back to OpenCV fps")
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Could not determine frame rate of {path}")
    if fps > 60:
        log.warning(
            "OpenCV reports %.1f fps, which is likely a bogus nominal rate. "
            "Install ffprobe for a reliable estimate.", fps,
        )
    return fps


class VideoReader:
    """Iterates frames of a video, subsampled to approximately target_fps."""

    def __init__(self, path: str | Path, target_fps: float | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.src_fps = probe_fps(self.path)
        self.stride = max(1, round(self.src_fps / target_fps)) if target_fps else 1
        self.sample_fps = self.src_fps / self.stride
        cap = cv2.VideoCapture(str(self.path))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    @property
    def sample_period(self) -> float:
        return 1.0 / self.sample_fps

    def __iter__(self) -> Iterator[FrameSample]:
        cap = cv2.VideoCapture(str(self.path))
        try:
            idx = 0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                if idx % self.stride == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        yield FrameSample(idx, idx / self.src_fps, frame)
                idx += 1
        finally:
            cap.release()
