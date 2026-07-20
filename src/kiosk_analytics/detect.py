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
        offset = np.zeros(4, dtype=np.float32)
        if self.cfg.roi is not None:
            rx1, ry1, rx2, ry2 = self.cfg.roi
            frame = frame[ry1:ry2, rx1:rx2]
            offset = np.array([rx1, ry1, rx1, ry1], dtype=np.float32)
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
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32) + offset
        conf = boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
        dets = np.hstack([xyxy, conf, cls])
        if self.cfg.nms_iou is not None and len(dets) > 1:
            dets = dets[_nms(dets[:, :4], dets[:, 4], self.cfg.nms_iou)]
        return dets


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS; returns indices of kept boxes. Used to deduplicate the
    output of NMS-free detectors (RT-DETR), whose duplicate boxes on one
    person otherwise spawn phantom tracks."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores)
    keep: list[int] = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return keep
