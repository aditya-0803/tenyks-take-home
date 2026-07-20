"""Multi-object tracker adapter over boxmot.

The online tracker's only job is to produce clean *tracklets*: short,
ID-switch-free fragments. Fragmentation (a person's track breaking when
they are occluded or leave the frame) is acceptable because the offline
stitching pass (stitch.py) re-links fragments into whole identities.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import TrackerCfg

log = logging.getLogger(__name__)

_TRACKER_CLASSES = {
    "bytetrack": "ByteTrack",
    "botsort": "BotSort",
    "ocsort": "OcSort",
    "strongsort": "StrongSort",
}

# Trackers that use appearance embeddings and need re-ID weights.
_NEEDS_REID = {"botsort", "strongsort"}


class Tracker:
    """Normalises boxmot trackers to: update(dets, frame) -> (M, 6).

    Output columns: x1, y1, x2, y2, track_id, conf.
    """

    def __init__(self, cfg: TrackerCfg, device: str = "cpu"):
        import boxmot

        name = cfg.type.lower()
        if name not in _TRACKER_CLASSES:
            raise ValueError(
                f"Unknown tracker '{cfg.type}'. Options: {sorted(_TRACKER_CLASSES)}"
            )
        tracker_cls = getattr(boxmot, _TRACKER_CLASSES[name])
        kwargs = dict(cfg.params)
        if name in _NEEDS_REID:
            kwargs.setdefault("reid_weights", Path(cfg.reid_weights))
            kwargs.setdefault("device", device)
            kwargs.setdefault("half", False)
        kwargs = self._filter_kwargs(tracker_cls, kwargs)
        try:
            self.impl = tracker_cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"boxmot {tracker_cls.__name__} rejected kwargs {kwargs}. "
                f"Check tracker.params against your boxmot version. ({e})"
            ) from e

    @staticmethod
    def _filter_kwargs(tracker_cls, kwargs: dict) -> dict:
        """Drop params the installed boxmot version doesn't accept (names
        drift between releases); warn so tuning intent isn't lost silently."""
        import inspect

        sig = inspect.signature(tracker_cls.__init__)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        dropped = sorted(set(kwargs) - set(accepted))
        if dropped:
            log.warning(
                "%s does not accept params %s in this boxmot version; ignoring. "
                "Available: %s",
                tracker_cls.__name__, dropped,
                sorted(k for k in sig.parameters if k != "self"),
            )
        return accepted

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if dets.size == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        out = self.impl.update(dets, frame)
        if out is None or len(out) == 0:
            return np.empty((0, 6), dtype=np.float32)
        out = np.asarray(out, dtype=np.float32)
        # boxmot returns [x1, y1, x2, y2, id, conf, cls, det_ind]
        return out[:, :6]
