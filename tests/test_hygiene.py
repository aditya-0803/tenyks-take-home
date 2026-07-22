import numpy as np

from kiosk_analytics.pipeline import _contamination_flags
from kiosk_analytics.tracklets import Tracklet


def det(x1, y1, x2, y2, conf=0.9):
    return [x1, y1, x2, y2, conf, 0]


def test_overlapping_detections_flagged():
    dets = np.array([
        det(100, 100, 180, 300),   # A
        det(150, 110, 230, 310),   # B overlaps A substantially
        det(600, 100, 680, 300),   # C isolated
    ])
    flags = _contamination_flags(dets)
    assert flags.tolist() == [True, True, False]


def test_single_detection_clean():
    dets = np.array([det(100, 100, 180, 300)])
    assert _contamination_flags(dets).tolist() == [False]


def test_contaminated_crops_are_fallback_only():
    """Clean crops always win; but a tracklet that lived its whole life in
    a crowd still gets (purified) fallback crops instead of none — an
    embedding-less tracklet can never be re-linked."""
    tr = Tracklet(1)
    frame = np.full((400, 400, 3), 128, dtype=np.uint8)
    box = np.array([100.0, 100.0, 180.0, 300.0])
    tr.add_crop_candidate(frame, box, conf=0.9, t=1.0, contaminated=True)
    assert len(tr.best_crops(8)) == 1  # fallback engaged
    tr.add_crop_candidate(frame, box, conf=0.9, t=2.0, contaminated=False)
    assert len(tr.best_crops(8)) == 1  # clean crop preferred, dirty ignored
