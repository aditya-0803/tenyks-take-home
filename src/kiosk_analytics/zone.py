"""Kiosk engagement zone: polygon membership tests for track anchor points."""

from __future__ import annotations

import numpy as np


class Zone:
    def __init__(self, polygon: list[list[float]]):
        if len(polygon) < 3:
            raise ValueError("zone_polygon needs at least 3 vertices")
        self.polygon = np.asarray(polygon, dtype=np.float64)

    def mask(self, height: int, width: int) -> np.ndarray:
        """Rasterised zone mask (bool, HxW). Polygon vertices may lie outside
        the frame (e.g. below the bottom edge to capture clipped boxes)."""
        import cv2

        m = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(m, [self.polygon.astype(np.int32)], 1)
        return m.astype(bool)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Vectorised ray-casting point-in-polygon test.

        points: (N, 2) array of (x, y). Returns boolean (N,).
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        x, y = pts[:, 0], pts[:, 1]
        poly = self.polygon
        inside = np.zeros(len(pts), dtype=bool)
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            crosses = (yi > y) != (yj > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_at_y = (xj - xi) * (y - yi) / (yj - yi) + xi
            inside ^= crosses & (x < x_at_y)
            j = i
        return inside


def anchor_points(boxes: np.ndarray, mode: str = "bottom_center") -> np.ndarray:
    """Representative ground point per box. boxes: (N, 4) xyxy -> (N, 2)."""
    boxes = np.atleast_2d(np.asarray(boxes, dtype=np.float64))
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    if mode == "bottom_center":
        return np.stack([cx, boxes[:, 3]], axis=1)
    if mode == "center":
        return np.stack([cx, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    raise ValueError(f"Unknown anchor mode '{mode}'")
