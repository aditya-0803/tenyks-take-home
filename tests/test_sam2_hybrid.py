"""Pure-logic tests for the sam2_hybrid engine (no torch/sam2 needed)."""

import numpy as np

from kiosk_analytics.sam2_hybrid import _iou, _mask_to_box, find_new_entrants


def box(x1, y1, x2, y2):
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def test_mask_to_box():
    m = np.zeros((100, 100), dtype=bool)
    m[20:60, 30:50] = True
    assert _mask_to_box(m).tolist() == [30, 20, 50, 60]
    assert _mask_to_box(np.zeros((10, 10), dtype=bool)) is None


def test_new_entrant_detected_after_min_frames():
    """A person appears at frame 2 and persists; no tracked mask explains
    them -> flagged at their first frame."""
    tracked = box(100, 100, 180, 300)  # existing person, tracked everywhere
    entrant = box(500, 120, 570, 320)
    dets = [
        np.stack([tracked]),                       # f0
        np.stack([tracked]),                       # f1
        np.stack([tracked, entrant]),              # f2 <- entrant appears
        np.stack([tracked, entrant + 3]),          # f3 (slight motion)
        np.stack([tracked, entrant + 6]),          # f4
    ]
    tracked_boxes = [[tracked]] * 5
    found = find_new_entrants(dets, tracked_boxes, [], iou_thresh=0.3, min_frames=3)
    assert found is not None
    frame_idx, boxes = found
    assert frame_idx == 2
    assert _iou(boxes[0], entrant) > 0.9


def test_flicker_not_flagged():
    """A one-frame unexplained detection (flicker) is ignored."""
    tracked = box(100, 100, 180, 300)
    ghost = box(500, 120, 570, 320)
    dets = [np.stack([tracked]), np.stack([tracked, ghost]), np.stack([tracked])]
    tracked_boxes = [[tracked]] * 3
    assert find_new_entrants(dets, tracked_boxes, [], 0.3, min_frames=2) is None


def test_already_prompted_not_reflagged():
    """If SAM2 was already prompted with this person and still produces no
    mask, don't loop forever re-prompting."""
    entrant = box(500, 120, 570, 320)
    dets = [np.stack([entrant])] * 4
    tracked_boxes = [[] for _ in range(4)]
    prompted = [(0, entrant)]
    assert find_new_entrants(dets, tracked_boxes, prompted, 0.3, min_frames=3) is None


def test_explained_detection_not_flagged():
    person = box(100, 100, 180, 300)
    dets = [np.stack([person])] * 4
    tracked_boxes = [[person + 2]] * 4  # mask box nearly identical
    assert find_new_entrants(dets, tracked_boxes, [], 0.3, min_frames=3) is None
