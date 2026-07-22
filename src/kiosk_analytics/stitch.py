"""Offline identity resolution: constrained agglomerative clustering of
tracklets.

The online tracker fragments each person into tracklets. Deciding which
tracklets belong to the same person is formulated as CLUSTERING under
hard physical constraints — not as successor matching. The previous
successor-based formulation (each tracklet links to at most one
follow-up) had a structural failure mode: a person fragmented into N
pieces needed N-1 consecutive pairwise wins, and one weak intermediate
fragment broke the chain even when a later fragment matched the first
one almost perfectly.

Formulation:
- Each tracklet gets an appearance embedding (L2-normalised mean of
  re-ID embeddings over its best crops, background masked out when
  segmentation masks are available).
- CANNOT-LINK constraints come from physics, not appearance:
  * significant temporal overlap  -> provably different people
    (brief overlap <= max_overlap_s is allowed: at an ID switch the
    dying track coasts while its replacement starts);
  * absence longer than max_gap_s -> treated as a different visit;
  * implied walking speed between exit and re-entry too high;
  * grossly different box heights.
- Agglomerative merging: repeatedly merge the closest pair of clusters
  (cosine distance between crop-count-weighted pooled embeddings) whose
  members all satisfy pairwise constraints, until no pair is closer than
  appearance_thresh. Pooling means every merge improves the cluster's
  embedding, so late fragments of a person can join even if some
  intermediate fragment was weak.

Known failure mode (documented for the write-up): visually identical
people who leave and return together may swap identities. The count
stays correct; the dwell error is bounded by the difference between
their durations.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import StitchCfg
from .tracklets import Tracklet, TrackletStore, split_tracklet

log = logging.getLogger(__name__)

INF = 1e6


# --------------------------------------------------------------------------
# Appearance embeddings
# --------------------------------------------------------------------------
class CropEmbedder:
    """Embeds person crops. Prefers a re-ID model (OSNet via boxmot); the
    generic torchvision fallback is opt-in because non-re-ID features make
    merges unreliable."""

    def __init__(
        self,
        backend: str = "auto",
        reid_weights: str = "osnet_x1_0_msmt17.pt",
        device: str = "cpu",
        allow_fallback: bool = False,
    ):
        self.device = device
        self._impl = None
        self._kind = None
        if backend in ("auto", "osnet"):
            try:
                self._impl = self._load_osnet(reid_weights)
                self._kind = "osnet"
            except Exception as e:  # noqa: BLE001
                if backend == "osnet" or not allow_fallback:
                    raise RuntimeError(
                        "Could not load the OSNet re-ID backend from boxmot "
                        f"({e}). Stitching with generic ImageNet features "
                        "produces unreliable merges, so it is disabled by "
                        "default; set stitch.allow_fallback: true to override, "
                        "or stitch.enabled: false to skip stitching."
                    ) from e
                log.warning("OSNet re-ID unavailable (%s); using torchvision fallback", e)
        if self._impl is None:
            self._impl = self._load_torchvision()
            self._kind = "torchvision"

    @property
    def kind(self) -> str:
        return self._kind

    @staticmethod
    def _find_reid_backend_cls():
        """Locate ReidAutoBackend across boxmot versions (the module has
        moved, e.g. appearance.reid_auto_backend -> appearance.reid.auto_backend)."""
        import importlib
        import pkgutil

        candidates = [
            "boxmot.appearance.reid.auto_backend",
            "boxmot.appearance.reid_auto_backend",
        ]
        try:
            appearance = importlib.import_module("boxmot.appearance")
            for m in pkgutil.walk_packages(appearance.__path__, appearance.__name__ + "."):
                if "auto_backend" in m.name and m.name not in candidates:
                    candidates.append(m.name)
        except Exception:  # noqa: BLE001
            pass
        for name in candidates:
            try:
                mod = importlib.import_module(name)
                if hasattr(mod, "ReidAutoBackend"):
                    log.info("Using re-ID backend from %s", name)
                    return mod.ReidAutoBackend
            except ImportError:
                continue
        raise ImportError(
            "ReidAutoBackend not found in boxmot; tried " + "; ".join(candidates)
        )

    def _load_osnet(self, weights: str):
        from pathlib import Path

        backend_cls = self._find_reid_backend_cls()
        rab = backend_cls(weights=Path(weights), device=self.device, half=False)
        return rab.model if hasattr(rab, "model") else rab.get_backend()

    def _load_torchvision(self):
        import torch
        import torchvision

        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        model.fc = torch.nn.Identity()
        model.eval().to(self.device)
        return model

    def embed_each(self, crops: list[np.ndarray]) -> np.ndarray | None:
        """crops: list of BGR uint8 arrays -> (n, d) L2-normalised rows."""
        if not crops:
            return None
        if self._kind == "osnet":
            feats = []
            for crop in crops:
                h, w = crop.shape[:2]
                box = np.array([[0, 0, w, h]], dtype=np.float32)
                f = self._impl.get_features(box, crop)
                feats.append(np.asarray(f).reshape(-1))
            feats = np.stack(feats)
        else:
            import torch

            batch = np.stack([c[:, :, ::-1] for c in crops]).astype(np.float32) / 255.0
            batch = (batch - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).float().to(self.device)
            with torch.no_grad():
                feats = self._impl(tensor).cpu().numpy()
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
        return feats

    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray | None:
        """crops -> single L2-normalised mean embedding."""
        feats = self.embed_each(crops)
        if feats is None:
            return None
        mean = feats.mean(axis=0)
        return mean / (np.linalg.norm(mean) + 1e-9)


# --------------------------------------------------------------------------
# Constraints and clustering (pure logic; unit-testable without torch)
# --------------------------------------------------------------------------
def pair_block_reason(a: Tracklet, b: Tracklet, cfg: StitchCfg) -> str | None:
    """Why tracklets a and b may NOT belong to the same person, or None if
    the pair passes all hard physical gates. Appearance is judged
    separately; recording the blocking gate in the debug output is what
    lets us distinguish 'embeddings failed' from 'physics forbade it'."""
    earlier, later = (a, b) if a.start <= b.start else (b, a)
    gap = later.start - earlier.end
    if gap < -cfg.max_overlap_s:  # long coexistence: provably different people
        return f"overlap {-gap:.1f}s"
    if gap > cfg.max_gap_s:
        return f"gap {gap:.1f}s"
    if gap >= 0:
        dist = float(np.linalg.norm(later.entry_point - earlier.exit_point))
        speed = dist / max(gap, 0.5)
        if speed > cfg.max_speed_px_s:
            return f"speed {speed:.0f}px/s"
    else:
        # Tolerated overlap: only a coasting duplicate of ONE person, which
        # coincides spatially. Two real people overlapping briefly do not.
        d = _overlap_mean_dist(earlier, later)
        if d is not None and d > cfg.overlap_max_dist_px:
            return f"overlap_apart {d:.0f}px"
    h_ratio = later.robust_height() / max(earlier.robust_height(), 1e-6)
    if not (cfg.min_height_ratio <= h_ratio <= cfg.max_height_ratio):
        return f"height_ratio {h_ratio:.2f}"
    return None


def pair_allowed(a: Tracklet, b: Tracklet, cfg: StitchCfg) -> bool:
    return pair_block_reason(a, b, cfg) is None


def _overlap_mean_dist(earlier: Tracklet, later: Tracklet) -> float | None:
    """Mean distance between the two tracklets' box centres during their
    temporal overlap window; None if either has no samples there."""
    t0, t1 = later.start, min(earlier.end, later.end)
    e_pts = [
        ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, t)
        for t, b in zip(earlier.times, earlier.boxes)
        if t0 <= t <= t1
    ]
    l_pts = [
        ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, t)
        for t, b in zip(later.times, later.boxes)
        if t0 <= t <= t1
    ]
    if not e_pts or not l_pts:
        return None
    dists = []
    for lx, ly, lt in l_pts:
        ex, ey, _ = min(e_pts, key=lambda p: abs(p[2] - lt))  # nearest in time
        dists.append(np.hypot(lx - ex, ly - ey))
    return float(np.mean(dists))


def _cosine_dist(u: np.ndarray, v: np.ndarray) -> float:
    return float(1.0 - np.dot(u, v))


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def continuity_links(tracklets: list[Tracklet], cfg: StitchCfg) -> list[tuple[int, int]]:
    """Hard merges on spatial continuity: tracklet B starting within
    continuity_max_gap_s of A's end with essentially coincident boxes is
    the same human (nobody teleports in <1s). This is what makes CHUNK
    BOUNDARIES invisible — no appearance evidence needed, so it works even
    when embeddings are starved in crowds. Greedy one-to-one by IoU."""
    candidates = []
    for a in tracklets:
        for b in tracklets:
            if a.tid == b.tid:
                continue
            gap = b.start - a.end
            if not (0.0 <= gap <= cfg.continuity_max_gap_s):
                continue
            iou = _box_iou(a.boxes[-1], b.boxes[0])
            if iou >= cfg.continuity_min_iou:
                candidates.append((iou, a.tid, b.tid))
    candidates.sort(reverse=True)
    links, used_pred, used_succ = [], set(), set()
    for iou, a_tid, b_tid in candidates:
        if a_tid in used_pred or b_tid in used_succ:
            continue
        links.append((a_tid, b_tid))
        used_pred.add(a_tid)
        used_succ.add(b_tid)
    return links


def find_chimera_split(
    times: list[float], feats: np.ndarray, thresh: float
) -> tuple[float, float] | None:
    """Detect a tracklet that contains TWO different people (identity theft
    at a crossing: the track ID survives a merged-box event but continues
    on the wrong person).

    Scans every temporal split point (>= 2 crops per side) and measures the
    cosine distance between the mean embeddings of the two sides. If the
    best split separates the tracklet by more than `thresh` — a distance on
    the scale of a DIFFERENT-person gap — returns (split_time, distance).
    A single person's halves stay far below thresh.
    """
    n = len(feats)
    if n < 4:
        return None
    best_d, best_k = -1.0, None
    for k in range(2, n - 1):
        a = feats[:k].mean(axis=0)
        b = feats[k:].mean(axis=0)
        a /= np.linalg.norm(a) + 1e-9
        b /= np.linalg.norm(b) + 1e-9
        d = _cosine_dist(a, b)
        if d > best_d:
            best_d, best_k = d, k
    if best_k is None or best_d <= thresh:
        return None
    return (times[best_k - 1] + times[best_k]) / 2.0, best_d


def cluster_tracklets(
    tracklets: list[Tracklet],
    embeddings: dict[int, np.ndarray | None],
    weights: dict[int, int],
    cfg: StitchCfg,
    forced_links: list[tuple[int, int]] | None = None,
) -> tuple[list[set[int]], dict]:
    """Constrained agglomerative clustering.

    Returns (clusters as sets of tids, debug info). forced_links (e.g.
    spatial-continuity chunk-boundary pairs) are merged unconditionally
    before appearance clustering begins. Then clusters merge greedily by
    smallest cosine distance between pooled embeddings, subject to every
    cross-pair of members passing pair_allowed, until no admissible pair
    is closer than the (gap-tiered) appearance threshold.
    """
    idx = {tr.tid: tr for tr in tracklets}
    allowed_cache: dict[tuple[int, int], bool] = {}

    def allowed(t1: int, t2: int) -> bool:
        key = (min(t1, t2), max(t1, t2))
        if key not in allowed_cache:
            allowed_cache[key] = pair_allowed(idx[t1], idx[t2], cfg)
        return allowed_cache[key]

    # Seed clusters from forced links via union-find.
    uf_parent = {tr.tid: tr.tid for tr in tracklets}

    def find(x):
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    for a, b in forced_links or []:
        if a in uf_parent and b in uf_parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                uf_parent[rb] = ra
    groups: dict[int, set[int]] = {}
    for tr in tracklets:
        groups.setdefault(find(tr.tid), set()).add(tr.tid)

    clusters: list[dict] = []
    for members in groups.values():
        embs = [
            (embeddings[t], max(weights.get(t, 1), 1))
            for t in members
            if embeddings.get(t) is not None
        ]
        if embs:
            pooled = sum(e * w for e, w in embs)
            pooled = pooled / (np.linalg.norm(pooled) + 1e-9)
            w_total = sum(w for _, w in embs)
        else:
            pooled, w_total = None, 1
        clusters.append({"members": set(members), "emb": pooled, "w": w_total})

    def merge_thresh(ci: dict, cj: dict) -> float:
        """Gap-tiered: short absences drift less in appearance."""
        min_gap = INF
        for t1 in ci["members"]:
            for t2 in cj["members"]:
                a, b = idx[t1], idx[t2]
                earlier, later = (a, b) if a.start <= b.start else (b, a)
                min_gap = min(min_gap, max(0.0, later.start - earlier.end))
        return (
            cfg.appearance_thresh_short
            if min_gap <= cfg.short_gap_s
            else cfg.appearance_thresh
        )

    debug_pairs = []
    for i, a in enumerate(tracklets):
        for b in tracklets[i + 1:]:
            ea, eb = embeddings.get(a.tid), embeddings.get(b.tid)
            if ea is None or eb is None:
                continue
            d = _cosine_dist(ea, eb)
            if d >= 0.6:  # keep the debug file readable
                continue
            entry = {"tid_a": a.tid, "tid_b": b.tid, "cost": round(d, 4)}
            reason = pair_block_reason(idx[a.tid], idx[b.tid], cfg)
            if reason is not None:
                # Appearance says "maybe same person" but physics forbids
                # the merge — exactly the entries to inspect when an ID
                # visibly splits after an occlusion.
                entry["blocked_by"] = reason
            debug_pairs.append(entry)

    merges = []
    while True:
        best = None
        for i in range(len(clusters)):
            ci = clusters[i]
            if ci["emb"] is None:
                continue
            for j in range(i + 1, len(clusters)):
                cj = clusters[j]
                if cj["emb"] is None:
                    continue
                d = _cosine_dist(ci["emb"], cj["emb"])
                if d >= merge_thresh(ci, cj):
                    continue
                if best is not None and d >= best[0]:
                    continue
                if all(
                    allowed(t1, t2) for t1 in ci["members"] for t2 in cj["members"]
                ):
                    best = (d, i, j)
        if best is None:
            break
        d, i, j = best
        ci, cj = clusters[i], clusters[j]
        merges.append(
            {
                "cost": round(d, 4),
                "into": sorted(ci["members"]),
                "absorbed": sorted(cj["members"]),
            }
        )
        pooled = ci["emb"] * ci["w"] + cj["emb"] * cj["w"]
        ci["emb"] = pooled / (np.linalg.norm(pooled) + 1e-9)
        ci["w"] += cj["w"]
        ci["members"] |= cj["members"]
        del clusters[j]

    debug = {"candidates": debug_pairs, "merges": merges}
    return [c["members"] for c in clusters], debug


def stitch_tracklets(
    store: TrackletStore, cfg: StitchCfg, device: str = "cpu"
) -> tuple[dict[int, int], dict]:
    """Returns (mapping tracker_id -> person_id, debug info).

    person_ids are 1-based, ordered by first appearance. With stitching
    disabled the mapping is a relabelling of tracker IDs."""
    tracklets = store.by_start_time()
    debug: dict = {"backend": None, "n_tracklets": len(tracklets), "chimera_splits": []}
    if cfg.enabled and len(tracklets) > 1:
        embedder = CropEmbedder(cfg.backend, cfg.reid_weights, device, cfg.allow_fallback)
        debug["backend"] = embedder.kind
        if embedder.kind != "osnet":
            log.warning(
                "STITCHING IS USING THE %s FALLBACK, NOT A RE-ID MODEL. "
                "appearance_thresh=%.2f was tuned for OSNet cosine distances "
                "and is likely meaningless here.", embedder.kind.upper(),
                cfg.appearance_thresh,
            )

        # ---- chimera pass: split tracklets that contain two people -------
        if cfg.chimera_thresh is not None:
            next_tid = max(store.tracklets) + 1
            for tid in sorted(store.tracklets):
                tr = store.tracklets[tid]
                timeline = tr.timeline_crops(cfg.chimera_max_crops)
                if len(timeline) < 4:
                    continue
                feats = embedder.embed_each([c for _, c in timeline])
                found = find_chimera_split(
                    [t for t, _ in timeline], feats, cfg.chimera_thresh
                )
                if found is None:
                    continue
                t_split, dist = found
                first, second = split_tracklet(tr, t_split, next_tid)
                if not first.times or not second.times:
                    continue
                store.tracklets[tid] = first
                store.tracklets[next_tid] = second
                debug["chimera_splits"].append(
                    {
                        "tid": tid,
                        "new_tid": next_tid,
                        "split_t": round(t_split, 1),
                        "halves_dist": round(dist, 4),
                    }
                )
                log.info(
                    "Chimera split: tracklet %d cut at t=%.1fs "
                    "(halves distance %.3f) -> new tracklet %d",
                    tid, t_split, dist, next_tid,
                )
                next_tid += 1
            tracklets = store.by_start_time()

        debug["n_tracklets"] = len(tracklets)
        debug["tracklets"] = [
            {
                "tid": tr.tid,
                "start": round(tr.start, 1),
                "end": round(tr.end, 1),
                "robust_height": round(tr.robust_height(), 1),
                "n_crops": len(tr.best_crops(cfg.crops_per_track)),
            }
            for tr in tracklets
        ]
        crops = {tr.tid: tr.best_crops(cfg.crops_per_track) for tr in tracklets}
        embeddings = {tid: embedder.embed_crops(c) for tid, c in crops.items()}
        weights = {tid: len(c) for tid, c in crops.items()}
        debug["tracklets_without_embedding"] = sorted(
            t for t, e in embeddings.items() if e is None
        )
        forced = continuity_links(tracklets, cfg)
        debug["continuity_links"] = [{"from_tid": a, "to_tid": b} for a, b in forced]
        log.info("Continuity links (chunk-boundary glue): %d", len(forced))
        clusters, cluster_debug = cluster_tracklets(
            tracklets, embeddings, weights, cfg, forced_links=forced
        )
        debug.update(cluster_debug)
        n_merged = sum(len(c) - 1 for c in clusters)
        log.info(
            "Clustering: %d tracklets -> %d identities (%d merges)",
            len(tracklets), len(clusters), n_merged,
        )
    else:
        clusters = [{tr.tid} for tr in tracklets]

    # Renumber clusters by first appearance.
    first_seen = {
        frozenset(c): min(store.tracklets[t].start for t in c) for c in clusters
    }
    ordered = sorted(first_seen, key=first_seen.get)
    mapping: dict[int, int] = {}
    for pid, members in enumerate(ordered, start=1):
        for tid in members:
            mapping[tid] = pid
    debug["tid_to_pid"] = {str(t): p for t, p in sorted(mapping.items())}
    return mapping, debug
