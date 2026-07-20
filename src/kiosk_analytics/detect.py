"""Person detection behind a single interface (ultralytics YOLO / RT-DETR)."""

from __future__ import annotations

import numpy as np

from .config import DetectorCfg

PERSON_CLASS = 0


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PersonDetector:
    """Wraps an ultralytics model; returns person detections as (N, 6) arrays.

    Output columns: x1, y1, x2, y2, conf, cls  (cls is always 0 / person).
    """

    def __init__(self, cfg: DetectorCfg):
        from ultralytics import RTDETR, YOLO

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        model_cls = RTDETR if "rtdetr" in cfg.model.lower() else YOLO
        self.model = model_cls(cfg.model)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        result = self.model.predict(
            frame,
            classes=[PERSON_CLASS],
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
        return np.hstack([xyxy, conf, cls])
