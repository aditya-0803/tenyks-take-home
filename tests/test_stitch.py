import numpy as np

from kiosk_analytics.config import StitchCfg
from kiosk_analytics.stitch import cluster_tracklets, pair_allowed
from kiosk_analytics.tracklets import Tracklet

CFG = StitchCfg(
    max_gap_s=100.0, max_overlap_s=1.0, appearance_thresh=0.3, max_speed_px_s=400.0
)


def make_tracklet(tid, t0, t1, x=500.0, y=600.0, h=200.0, n=5):
    tr = Tracklet(tid)
    for i, t in enumerate(np.linspace(t0, t1, n)):
        box = np.array([x - 40, y - h, x + 40, y])
        tr.add(i, float(t), box, 0.9)
    return tr


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def run_cluster(tracklets, embs):
    weights = {t.tid: 5 for t in tracklets}
    clusters, debug = cluster_tracklets(tracklets, embs, weights, CFG)
    return {frozenset(c) for c in clusters}, debug


# ---- constraint gates -----------------------------------------------------
def test_long_overlap_forbidden():
    a = make_tracklet(1, 0, 50)
    b = make_tracklet(2, 40, 90)  # 10s coexistence: different people
    assert not pair_allowed(a, b, CFG)


def test_brief_overlap_from_id_switch_allowed():
    a = make_tracklet(1, 0, 50.0)
    b = make_tracklet(2, 49.5, 90)  # 0.5s < max_overlap_s
    assert pair_allowed(a, b, CFG)


def test_gap_and_speed_gates():
    a = make_tracklet(1, 0, 50)
    assert not pair_allowed(a, make_tracklet(2, 200, 250), CFG)  # gap 150 > 100
    assert not pair_allowed(a, make_tracklet(3, 50.5, 90, x=5000.0), CFG)  # teleport


def test_gate_is_symmetric():
    a = make_tracklet(1, 0, 50)
    b = make_tracklet(2, 60, 90)
    assert pair_allowed(a, b, CFG) == pair_allowed(b, a, CFG)


# ---- clustering -----------------------------------------------------------
def test_fragmented_person_unites_without_chain():
    """The structural failure of successor matching: person fragments into
    A (long), B (short, noisy embedding), C (long, clean). Clustering must
    unite all three; a successor chain would break at B."""
    a = make_tracklet(1, 0, 50)
    b = make_tracklet(2, 52, 58)
    c = make_tracklet(3, 90, 150)
    e_person = unit([1, 0, 0])
    embs = {1: e_person, 2: unit([1, 0.35, 0]), 3: unit([1, 0.05, 0])}
    clusters, _ = run_cluster([a, b, c], embs)
    assert clusters == {frozenset({1, 2, 3})}


def test_cooccurring_lookalike_stays_separate():
    """Same appearance but temporally overlapping: physics wins over
    appearance; must remain two people."""
    a = make_tracklet(1, 0, 145)
    d = make_tracklet(2, 100, 160)  # 45s coexistence with a
    e = unit([1, 0, 0])
    clusters, _ = run_cluster([a, d], {1: e, 2: e})
    assert clusters == {frozenset({1}), frozenset({2})}


def test_constraint_propagates_through_cluster():
    """If B co-occurs with A, then a cluster containing A can never absorb
    B, even via an intermediate C that is compatible with both."""
    a = make_tracklet(1, 0, 60)
    b = make_tracklet(2, 30, 100)   # overlaps a -> cannot ever join a
    c = make_tracklet(3, 110, 150)  # compatible with both a and b
    e = unit([1, 0, 0])
    clusters, _ = run_cluster([a, b, c], {1: e, 2: e, 3: e})
    # c joins exactly one of a/b; a and b never share a cluster
    for cl in clusters:
        assert not ({1, 2} <= cl)


def test_group_leave_and_return_resolved_jointly():
    a1 = make_tracklet(1, 0, 60, x=500)
    a2 = make_tracklet(2, 0, 60, x=600)
    b1 = make_tracklet(3, 120, 180, x=520)
    b2 = make_tracklet(4, 120, 180, x=580)
    e_red, e_blue = unit([1, 0.1, 0]), unit([0, 1, 0.1])
    embs = {1: e_red, 2: e_blue, 3: e_red, 4: e_blue}
    clusters, debug = run_cluster([a1, a2, b1, b2], embs)
    assert clusters == {frozenset({1, 3}), frozenset({2, 4})}
    assert len(debug["merges"]) == 2


def test_new_arrival_not_absorbed():
    a = make_tracklet(1, 0, 60)
    b = make_tracklet(2, 100, 160)
    embs = {1: unit([1, 0, 0]), 2: unit([0, 1, 0])}  # distance 1.0 > 0.3
    clusters, _ = run_cluster([a, b], embs)
    assert clusters == {frozenset({1}), frozenset({2})}


def test_missing_embedding_stays_singleton():
    a = make_tracklet(1, 0, 60)
    b = make_tracklet(2, 100, 160)
    clusters, _ = run_cluster([a, b], {1: unit([1, 0, 0]), 2: None})
    assert clusters == {frozenset({1}), frozenset({2})}
