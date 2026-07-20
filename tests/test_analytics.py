import numpy as np
import pytest

from kiosk_analytics.analytics import (
    _debounce,
    compute_person_results,
    merge_segments,
)
from kiosk_analytics.config import AnalyticsCfg
from kiosk_analytics.zone import Zone, anchor_points

SQUARE = Zone([[0, 0], [100, 0], [100, 100], [0, 100]])


def make_identity(times, xs, y=50.0, box_h=40.0):
    """Identity dict with bottom-centre anchors at (x, y)."""
    boxes = np.array([[x - 10, y - box_h, x + 10, y] for x in xs])
    return {
        "times": np.asarray(times, dtype=float),
        "frame_indices": np.arange(len(times)),
        "boxes": boxes,
        "source_tids": [1],
    }


def test_zone_contains():
    pts = np.array([[50, 50], [150, 50], [-1, 50], [99.5, 99.5]])
    assert SQUARE.contains(pts).tolist() == [True, False, False, True]


def test_anchor_points():
    boxes = np.array([[0, 0, 20, 40]])
    assert anchor_points(boxes, "bottom_center").tolist() == [[10, 40]]
    assert anchor_points(boxes, "center").tolist() == [[10, 20]]


def test_debounce_kills_flicker():
    raw = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
    out = _debounce(raw, k=2)
    # single-sample blips ignored; state flips on the 2nd consecutive
    # opposite sample (index 6), then the lone 1 at index 8 is ignored
    assert out.tolist() == [True] * 6 + [False] * 6


def test_merge_segments():
    segs = [(0, 10), (12, 20), (30, 40)]
    assert merge_segments(segs, gap=3) == [(0, 20), (30, 40)]
    assert merge_segments(segs, gap=1) == segs


def test_dwell_policy_a_excludes_gap():
    """Person in zone 0-20s, out 20-50s, back in 50-60s => dwell 30s."""
    times = np.arange(0, 61, 1.0)
    xs = [50 if (t <= 20 or t >= 50) else 200 for t in times]
    identity = {1: make_identity(times, xs)}
    cfg = AnalyticsCfg(hysteresis_samples=1, merge_gap_s=2.0, min_engagement_s=8.0)
    results, _ = compute_person_results(identity, SQUARE, cfg, sample_period=1.0)
    assert len(results) == 1
    r = results[0]
    assert r.engaged
    assert len(r.segments) == 2
    assert r.dwell_s == pytest.approx(30.0, abs=2.0)


def test_walkthrough_not_engaged():
    times = np.arange(0, 30, 1.0)
    xs = [50 if t < 4 else 300 for t in times]  # only 4s in zone
    identity = {1: make_identity(times, xs)}
    cfg = AnalyticsCfg(hysteresis_samples=1, min_engagement_s=8.0)
    results, _ = compute_person_results(identity, SQUARE, cfg, sample_period=1.0)
    assert len(results) == 1
    assert not results[0].engaged


def test_undetected_gap_not_credited():
    """Samples 0-10s then 100-110s (tracker lost them in between):
    the 90s hole must not count as dwell."""
    times = np.concatenate([np.arange(0, 11, 1.0), np.arange(100, 111, 1.0)])
    xs = [50] * len(times)
    identity = {1: make_identity(times, xs)}
    cfg = AnalyticsCfg(hysteresis_samples=1, merge_gap_s=2.0, min_engagement_s=8.0)
    results, _ = compute_person_results(identity, SQUARE, cfg, sample_period=1.0)
    assert results[0].dwell_s == pytest.approx(20.0, abs=2.0)


def test_short_track_dropped():
    identity = {1: make_identity([0.0, 0.5], [50, 50])}
    cfg = AnalyticsCfg(min_track_len_s=1.0)
    results, _ = compute_person_results(identity, SQUARE, cfg, sample_period=0.5)
    assert results == []


def test_bottom_strip_handles_frame_clipped_box():
    """Box clipped by the frame bottom edge: the single-point anchor lands
    exactly on the polygon boundary and tests OUTSIDE (the bug observed on
    the first Colab run), while bottom_strip membership still counts it."""
    frame_w, frame_h = 200, 200
    zone = Zone([[0, 100], [200, 100], [200, 220], [0, 220]])  # extends past frame
    times = np.arange(0, 20, 1.0)
    # box bottom pinned to the frame edge (y2 = 200)
    boxes = np.array([[80, 120, 120, 200]] * len(times), dtype=float)
    identity = {
        1: {
            "times": times,
            "frame_indices": np.arange(len(times)),
            "boxes": boxes,
            "source_tids": [1],
        }
    }
    anchor_cfg = AnalyticsCfg(membership="anchor", hysteresis_samples=1, min_engagement_s=8.0)
    zone_no_overhang = Zone([[0, 100], [200, 100], [200, 200], [0, 200]])
    results_anchor, _ = compute_person_results(
        identity, zone_no_overhang, anchor_cfg, 1.0, frame_size=(frame_w, frame_h)
    )
    assert not results_anchor[0].engaged  # demonstrates the failure mode

    strip_cfg = AnalyticsCfg(membership="bottom_strip", hysteresis_samples=1, min_engagement_s=8.0)
    results_strip, _ = compute_person_results(
        identity, zone, strip_cfg, 1.0, frame_size=(frame_w, frame_h)
    )
    assert results_strip[0].engaged
    assert results_strip[0].dwell_s == pytest.approx(19.0, abs=1.0)
