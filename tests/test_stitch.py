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
    """Coasting duplicate: same place during the overlap -> mergeable."""
    a = make_tracklet(1, 0, 50.0, x=500, n=101)
    b = make_tracklet(2, 49.5, 90, x=505, n=82)  # 0.5s overlap, coincident
    assert pair_allowed(a, b, CFG)


def test_overlapping_people_apart_stay_split():
    """Two REAL people whose fragments overlap briefly stand apart during
    the overlap -> merge forbidden even with identical appearance. (The
    two-people-one-ID regression from raising max_overlap_s.)"""
    cfg = StitchCfg(
        max_gap_s=100.0, max_overlap_s=5.0,
        appearance_thresh=0.3, max_speed_px_s=400.0, overlap_max_dist_px=120.0,
    )
    a = make_tracklet(1, 0, 52, x=500, n=105)     # woman at kiosk
    b = make_tracklet(2, 49, 90, x=900, n=83)     # man walking, 3s overlap
    assert not pair_allowed(a, b, cfg)
    e = unit([1, 0, 0])
    weights = {1: 5, 2: 5}
    clusters, _ = cluster_tracklets([a, b], {1: e, 2: e}, weights, cfg)
    assert {frozenset(c) for c in clusters} == {frozenset({1}), frozenset({2})}


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


# ---- continuity links (chunk-boundary glue) -------------------------------
def test_chunk_boundary_glued_without_appearance():
    """Fragments split by a chunk boundary (gap ~0.1s, coincident boxes)
    must merge even with NO embeddings — spatial continuity is proof."""
    from kiosk_analytics.stitch import continuity_links

    a = make_tracklet(1, 0, 45.0, x=500)
    b = make_tracklet(2, 45.1, 90, x=502)   # next chunk, same spot
    c = make_tracklet(3, 45.1, 90, x=900)   # different person, far away
    links = continuity_links([a, b, c], CFG)
    assert links == [(1, 2)]

    clusters, _ = cluster_tracklets(
        [a, b, c], {1: None, 2: None, 3: None}, {}, CFG, forced_links=links
    )
    assert {frozenset(cl) for cl in clusters} == {frozenset({1, 2}), frozenset({3})}


def test_continuity_one_to_one():
    """Two candidates after one ender: only the best-IoU pairing links."""
    from kiosk_analytics.stitch import continuity_links

    a = make_tracklet(1, 0, 45.0, x=500)
    b = make_tracklet(2, 45.1, 90, x=501)   # near-perfect continuation
    c = make_tracklet(3, 45.1, 90, x=530)   # overlaps somewhat, worse IoU
    links = continuity_links([a, b, c], CFG)
    assert (1, 2) in links
    assert all(link[1] != 3 or link[0] != 1 for link in links)


def test_gap_tiered_threshold_allows_short_absence_rematch():
    """Cost 0.30 merge: rejected across a long gap (> short_gap_s), accepted
    across a short absence."""
    cfg = StitchCfg(
        max_gap_s=100.0, appearance_thresh=0.26, appearance_thresh_short=0.34,
        short_gap_s=10.0, max_speed_px_s=400.0,
    )
    e1, e2 = unit([1, 0, 0]), unit([1, 1, 0])  # cosine dist ~0.293
    d = 1 - float(np.dot(e1, e2))
    assert 0.26 < d < 0.34

    short = [make_tracklet(1, 0, 50), make_tracklet(2, 55, 90)]     # 5s gap
    clusters, _ = cluster_tracklets(short, {1: e1, 2: e2}, {}, cfg)
    assert {frozenset(c) for c in clusters} == {frozenset({1, 2})}

    long_ = [make_tracklet(1, 0, 50), make_tracklet(2, 80, 120)]    # 30s gap
    clusters, _ = cluster_tracklets(long_, {1: e1, 2: e2}, {}, cfg)
    assert {frozenset(c) for c in clusters} == {frozenset({1}), frozenset({2})}


# ---- chimera detection ----------------------------------------------------
def test_chimera_split_detects_identity_theft():
    """Tracklet whose first 4 crops are person A and last 6 are person B
    (merged-box ID theft at a crossing) -> split lands at the boundary."""
    from kiosk_analytics.stitch import find_chimera_split

    e_a, e_b = unit([1, 0.05, 0]), unit([0, 1, 0.05])
    feats = np.stack([e_a] * 4 + [e_b] * 6)
    times = list(np.arange(10.5, 20.5, 1.0))
    found = find_chimera_split(times, feats, thresh=0.35)
    assert found is not None
    t_split, dist = found
    assert 13.5 < t_split < 15.5  # between 4th and 5th crop
    assert dist > 0.8


def test_no_chimera_split_for_single_person():
    from kiosk_analytics.stitch import find_chimera_split

    rng = np.random.default_rng(0)
    base = unit([1, 0.2, 0.1])
    feats = np.stack([unit(base + rng.normal(0, 0.05, 3)) for _ in range(10)])
    assert find_chimera_split(list(range(10)), feats, thresh=0.35) is None


def test_split_tracklet_partitions_observations():
    from kiosk_analytics.tracklets import split_tracklet

    tr = make_tracklet(7, 0, 90, n=10)  # samples at t=0,10,...,90
    first, second = split_tracklet(tr, t_split=45.0, new_tid=1007)
    assert first.tid == 7 and second.tid == 1007
    assert len(first.times) == 5 and len(second.times) == 5
    assert first.end < 45.0 <= second.start


def test_missing_embedding_stays_singleton():
    a = make_tracklet(1, 0, 60)
    b = make_tracklet(2, 100, 160)
    clusters, _ = run_cluster([a, b], {1: unit([1, 0, 0]), 2: None})
    assert clusters == {frozenset({1}), frozenset({2})}
