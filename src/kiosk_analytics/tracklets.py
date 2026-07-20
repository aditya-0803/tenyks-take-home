"""Tracklet storage: per-track observations plus best crops for re-ID."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

CROP_SIZE = (128, 256)  # (w, h) fed to the re-ID embedder
CROP_BIN_S = 1.0        # keep at most one candidate crop per second of track


@dataclass
class Tracklet:
    tid: int
    times: list[float] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)
    confs: list[float] = field(default_factory=list)
    # time-bin -> (quality score, crop); winnowed to top-k at read time
    _crops: dict[int, tuple[float, np.ndarray]] = field(default_factory=dict)

    def add(self, frame_idx: int, t: float, box: np.ndarray, conf: float) -> None:
        self.frame_indices.append(frame_idx)
        self.times.append(t)
        self.boxes.append(box.astype(np.float32))
        self.confs.append(float(conf))

    def add_crop_candidate(self, frame: np.ndarray, box: np.ndarray, conf: float, t: float) -> None:
        """Keep the highest-quality crop per one-second bin of the track.

        Quality = detection confidence * sqrt(box area): prefers large,
        confident (unoccluded) views of the person.
        """
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        if x2 - x1 < 10 or y2 - y1 < 20:
            return
        score = conf * np.sqrt((x2 - x1) * (y2 - y1))
        bin_idx = int(t / CROP_BIN_S)
        prev = self._crops.get(bin_idx)
        if prev is None or score > prev[0]:
            crop = cv2.resize(frame[y1:y2, x1:x2], CROP_SIZE)
            self._crops[bin_idx] = (score, crop)

    def best_crops(self, k: int) -> list[np.ndarray]:
        ranked = sorted(self._crops.values(), key=lambda sc: -sc[0])
        return [crop for _, crop in ranked[:k]]

    # --- geometry/timing helpers used by stitching -------------------------
    @property
    def start(self) -> float:
        return self.times[0]

    @property
    def end(self) -> float:
        return self.times[-1]

    @property
    def duration(self) -> float:
        return self.end - self.start

    def _anchor(self, box: np.ndarray) -> np.ndarray:
        return np.array([(box[0] + box[2]) / 2, box[3]])

    @property
    def entry_point(self) -> np.ndarray:
        return self._anchor(self.boxes[0])

    @property
    def exit_point(self) -> np.ndarray:
        return self._anchor(self.boxes[-1])

    def mean_height(self, n: int = 5) -> float:
        hs = [b[3] - b[1] for b in self.boxes[-n:]]
        return float(np.mean(hs))


class TrackletStore:
    def __init__(self):
        self.tracklets: dict[int, Tracklet] = {}

    def add_observation(
        self,
        tid: int,
        frame_idx: int,
        t: float,
        box: np.ndarray,
        conf: float,
        frame: np.ndarray,
    ) -> None:
        tr = self.tracklets.get(tid)
        if tr is None:
            tr = self.tracklets[tid] = Tracklet(tid)
        tr.add(frame_idx, t, box, conf)
        tr.add_crop_candidate(frame, box, conf, t)

    def __len__(self) -> int:
        return len(self.tracklets)

    def by_start_time(self) -> list[Tracklet]:
        return sorted(self.tracklets.values(), key=lambda tr: tr.start)
