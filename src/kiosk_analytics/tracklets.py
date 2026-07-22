"""Tracklet storage: per-track observations plus best crops for re-ID."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

CROP_SIZE = (128, 256)  # (w, h) fed to the re-ID embedder
CROP_BIN_S = 1.0        # keep at most one candidate crop per second of track


def _letterbox(crop: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize preserving aspect ratio, padding with black. A stretched
    torso-only crop (feet occluded by the kiosk) otherwise distorts the
    re-ID embedding and inflates same-person distances."""
    w, h = size
    ch, cw = crop.shape[:2]
    scale = min(w / cw, h / ch)
    nw, nh = max(int(cw * scale), 1), max(int(ch * scale), 1)
    resized = cv2.resize(crop, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=crop.dtype)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


@dataclass
class Tracklet:
    tid: int
    times: list[float] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)
    confs: list[float] = field(default_factory=list)
    # time-bin -> (quality score, crop); winnowed to top-k at read time
    _crops: dict[int, tuple[float, np.ndarray]] = field(default_factory=dict)
    # Fallback bin: crops from frames where the box overlapped another
    # detection. Masks purify these substantially, so they are usable when
    # a tracklet spends its whole life in a crowd — a weak embedding beats
    # a missing one (embedding-less tracklets can never be re-linked).
    _crops_dirty: dict[int, tuple[float, np.ndarray]] = field(default_factory=dict)

    def add(self, frame_idx: int, t: float, box: np.ndarray, conf: float) -> None:
        self.frame_indices.append(frame_idx)
        self.times.append(t)
        self.boxes.append(box.astype(np.float32))
        self.confs.append(float(conf))

    def add_crop_candidate(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        conf: float,
        t: float,
        mask: np.ndarray | None = None,
        contaminated: bool = False,
    ) -> None:
        """Keep the highest-quality crop per one-second bin of the track.

        Quality = detection confidence * sqrt(box area): prefers large,
        confident (unoccluded) views of the person. When a segmentation
        mask is provided, background/neighbour pixels are zeroed so the
        re-ID embedding sees only this person (critical in queues, where
        a box routinely contains half of the adjacent person).

        Frames flagged `contaminated` (box overlapping another detection)
        go to a fallback store: clean crops are always preferred for
        embeddings, but a tracklet that lived its whole life in a crowd
        still gets a (mask-purified) embedding instead of none at all.
        """
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        if x2 - x1 < 10 or y2 - y1 < 20:
            return
        score = conf * np.sqrt((x2 - x1) * (y2 - y1))
        bin_idx = int(t / CROP_BIN_S)
        target = self._crops_dirty if contaminated else self._crops
        prev = target.get(bin_idx)
        if prev is None or score > prev[0]:
            crop = frame[y1:y2, x1:x2]
            if mask is not None:
                crop = crop * mask[y1:y2, x1:x2, None].astype(crop.dtype)
            target[bin_idx] = (score, _letterbox(crop, CROP_SIZE))

    def best_crops(self, k: int) -> list[np.ndarray]:
        ranked = sorted(self._crops.values(), key=lambda sc: -sc[0])
        if not ranked:  # crowd-only tracklet: fall back to purified dirty crops
            ranked = sorted(self._crops_dirty.values(), key=lambda sc: -sc[0])
        return [crop for _, crop in ranked[:k]]

    def timeline_crops(self, max_n: int = 40) -> list[tuple[float, np.ndarray]]:
        """All kept crops in time order as (timestamp, crop), evenly
        subsampled to max_n. Used for within-tracklet appearance
        consistency checks (chimera detection). Falls back to including
        dirty crops when clean ones are too sparse."""
        source = self._crops
        if len(source) < 4:
            source = {**self._crops_dirty, **self._crops}  # clean wins bins
        entries = sorted(source.items())
        items = [
            (bin_idx * CROP_BIN_S + CROP_BIN_S / 2, crop)
            for bin_idx, (_, crop) in entries
        ]
        if len(items) > max_n:
            idx = np.linspace(0, len(items) - 1, max_n).astype(int)
            items = [items[i] for i in idx]
        return items

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

    def robust_height(self) -> float:
        """75th-percentile box height over the whole tracklet: measures the
        person's true size even when many frames are occlusion-clipped
        (endpoint heights are corrupted exactly when tracks break)."""
        hs = [b[3] - b[1] for b in self.boxes]
        return float(np.percentile(hs, 75))


def split_tracklet(tr: Tracklet, t_split: float, new_tid: int) -> tuple[Tracklet, Tracklet]:
    """Partition a tracklet at t_split into (before, after). The first half
    keeps the original tid; the second half gets new_tid. Used when a
    tracklet is found to contain two different people (identity theft at
    a crossing: the track continues, but on the wrong person)."""
    a, b = Tracklet(tr.tid), Tracklet(new_tid)
    for f, t, box, conf in zip(tr.frame_indices, tr.times, tr.boxes, tr.confs):
        target = a if t < t_split else b
        target.add(f, t, box, conf)
    for bin_idx, entry in tr._crops.items():
        t = bin_idx * CROP_BIN_S + CROP_BIN_S / 2
        target = a if t < t_split else b
        target._crops[bin_idx] = entry
    for bin_idx, entry in tr._crops_dirty.items():
        t = bin_idx * CROP_BIN_S + CROP_BIN_S / 2
        target = a if t < t_split else b
        target._crops_dirty[bin_idx] = entry
    return a, b


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
        mask: np.ndarray | None = None,
        contaminated: bool = False,
    ) -> None:
        tr = self.tracklets.get(tid)
        if tr is None:
            tr = self.tracklets[tid] = Tracklet(tid)
        tr.add(frame_idx, t, box, conf)
        tr.add_crop_candidate(frame, box, conf, t, mask=mask, contaminated=contaminated)

    def __len__(self) -> int:
        return len(self.tracklets)

    def by_start_time(self) -> list[Tracklet]:
        return sorted(self.tracklets.values(), key=lambda tr: tr.start)
