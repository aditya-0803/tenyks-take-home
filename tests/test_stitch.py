import numpy as np

from kiosk_analytics.config import StitchCfg
from kiosk_analytics.stitch import _UnionFind, assign_links, link_cost
from kiosk_analytics.tracklets import Tracklet

CFG = StitchCfg(max_gap_s=100.0, appearance_thresh=0.4, max_speed_px_s=400.0)


def make_tracklet(tid, t0, t1, x=500.0, y=600.0, h=200.0, n=5):
    tr = Tracklet(tid)
    for i, t in enumerate(np.linspace(t0, t1, n)):
        box = np.array([x - 40, y - h, x + 40, y])
        tr.add(i, float(t), box, 0.9)
    return tr


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_temporal_overlap_is_infinite_cost():
    a = make_tracklet(1, 0, 50)
    b = make_tracklet(2, 40, 90)  # overlaps a
    e = unit([1, 0, 0])
    assert link_cost(a, b, e, e, CFG) >= 1e6


def test_gap_and_speed_gates():
    a = make_tracklet(1, 0, 50)
    too_late = make_tracklet(2, 200, 250)  # gap 150 > max_gap 100
    e = unit([1, 0, 0])
    assert link_cost(a, too_late, e, e, CFG) >= 1e6

    teleport = make_tracklet(3, 50.5, 90, x=5000.0)  # 4500px in 0.5s
    assert link_cost(a, teleport, e, e, CFG) >= 1e6


def test_simultaneous_group_resolved_jointly():
    """Two people leave together and return together; joint assignment
    should link each to their own reappearance via appearance."""
    a1 = make_tracklet(1, 0, 60, x=500)
    a2 = make_tracklet(2, 0, 60, x=600)
    b1 = make_tracklet(3, 120, 180, x=520)
    b2 = make_tracklet(4, 120, 180, x=580)
    e_red, e_blue = unit([1, 0.1, 0]), unit([0, 1, 0.1])
    embs = {1: e_red, 2: e_blue, 3: e_red, 4: e_blue}
    links, candidates = assign_links([a1, a2, b1, b2], embs, CFG)
    assert set(links) == {(1, 3), (2, 4)}
    assert any(c["accepted"] for c in candidates)


def test_new_arrival_not_force_matched():
    """A person leaves; a different-looking person arrives later.
    No link should be made."""
    a = make_tracklet(1, 0, 60)
    b = make_tracklet(2, 100, 160)
    embs = {1: unit([1, 0, 0]), 2: unit([0, 1, 0])}  # cosine dist 1.0 > 0.4
    links, _ = assign_links([a, b], embs, CFG)
    assert links == []


def test_chain_merging_via_union_find():
    uf = _UnionFind([1, 2, 3])
    uf.union(1, 2)
    uf.union(2, 3)
    assert uf.find(3) == uf.find(1)


def test_missing_embedding_blocks_link():
    a = make_tracklet(1, 0, 60)
    b = make_tracklet(2, 100, 160)
    assert link_cost(a, b, None, unit([1, 0, 0]), CFG) >= 1e6
