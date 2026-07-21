import numpy as np

from kiosk_analytics.detect import _nms, _suppress_contained
from kiosk_analytics.tracklets import Tracklet


def test_torso_box_inside_body_box_suppressed():
    """A half-height torso box inside a full-body box has IoU ~0.5 (survives
    NMS) but containment ~1.0 (killed by IoM suppression). This duplicate
    caused the pid5/pid11 split: a phantom parallel track poisoned the
    identity cluster."""
    body = [100, 100, 180, 300]
    torso = [105, 105, 175, 200]
    boxes = np.array([body, torso], dtype=float)
    scores = np.array([0.9, 0.4])
    assert _nms(boxes, scores, iou_thresh=0.7) == [0, 1]  # NMS keeps both
    assert _suppress_contained(boxes, scores, iom_thresh=0.8) == [0]


def test_wide_box_veto_logic():
    """A merged two-person box (w/h ~0.8) is vetoed; single people and the
    toddler (w/h ~0.5-0.6) survive."""
    boxes = np.array([
        [100, 100, 180, 300],   # adult: 80/200 = 0.40
        [400, 200, 460, 300],   # child: 60/100 = 0.60
        [600, 100, 930, 500],   # merged pair: 330/400 = 0.83
    ], dtype=float)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w / h <= 0.7)
    assert keep.tolist() == [True, True, False]


def test_adjacent_people_not_suppressed():
    a = [100, 100, 180, 300]
    b = [170, 105, 250, 305]  # neighbour, slight overlap
    boxes = np.array([a, b], dtype=float)
    scores = np.array([0.9, 0.85])
    assert _suppress_contained(boxes, scores, iom_thresh=0.8) == [0, 1]


def test_robust_height_ignores_occlusion_clipped_frames():
    tr = Tracklet(1)
    # 12 clean frames at height 200, 8 occlusion-clipped at 100
    for i, h in enumerate([200] * 12 + [100] * 8):
        tr.add(i, float(i), np.array([100, 300 - h, 180, 300]), 0.9)
    assert tr.robust_height() == 200.0
    # endpoint-based mean is corrupted by the clipped tail
    assert tr.mean_height() == 100.0
