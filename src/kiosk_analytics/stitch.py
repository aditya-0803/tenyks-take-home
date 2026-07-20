"""Offline track stitching: re-link tracklet fragments into identities.

The online tracker breaks a person's track when they are occluded or
leave the frame. This pass treats re-linking as a GLOBAL assignment
problem so that groups who leave and return together are resolved
jointly rather than one at a time:

1. Each tracklet gets an appearance embedding: the L2-normalised mean of
   re-ID embeddings over its best crops (large, confident detections).
2. Candidate links (A -> B, where B starts after A ends) are gated by
   hard constraints: temporal overlap is impossible, gap <= max_gap_s,
   exit->entry displacement implies a plausible walking speed, and box
   heights are similar.
3. Surviving pairs form a cost matrix (cosine distance); the Hungarian
   algorithm picks the globally optimal one-to-one linking, with a
   per-link cost threshold acting as a "no match" option so new arrivals
   are not force-matched to departed tracklets.
4. Union-find merges accepted links; chains (A->B->C) emerge naturally.

Known failure mode (documented for the write-up): visually identical
people who leave and return together may swap IDs. This keeps the count
correct and typically produces small dwell error.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import StitchCfg
from .tracklets import Tracklet, TrackletStore

log = logging.getLogger(__name__)

INF = 1e6


# --------------------------------------------------------------------------
# Appearance embeddings
# --------------------------------------------------------------------------
class CropEmbedder:
    """Embeds person crops. Prefers a re-ID model (OSNet via boxmot); falls
    back to torchvision ResNet-18 features if boxmot's re-ID stack is
    unavailable."""

    def __init__(
        self,
        backend: str = "auto",
        reid_weights: str = "osnet_x0_25_msmt17.pt",
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
        errors = []
        for name in candidates:
            try:
                mod = importlib.import_module(name)
                if hasattr(mod, "ReidAutoBackend"):
                    log.info("Using re-ID backend from %s", name)
                    return mod.ReidAutoBackend
            except ImportError as e:
                errors.append(f"{name}: {e}")
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

    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray | None:
        """crops: list of BGR uint8 arrays -> L2-normalised mean embedding."""
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
        mean = feats.mean(axis=0)
        return mean / (np.linalg.norm(mean) + 1e-9)


# --------------------------------------------------------------------------
# Gating and assignment (pure logic; unit-testable without torch)
# --------------------------------------------------------------------------
def link_cost(a: Tracklet, b: Tracklet, emb_a, emb_b, cfg: StitchCfg) -> float:
    """Cost of linking tracklet A -> B (B is the same person reappearing).

    Returns INF if any hard gate fails, else cosine distance between
    appearance embeddings.
    """
    gap = b.start - a.end
    if gap <= 0 or gap > cfg.max_gap_s:
        return INF
    dist = float(np.linalg.norm(b.entry_point - a.exit_point))
    if dist / max(gap, 0.5) > cfg.max_speed_px_s:
        return INF
    h_ratio = b.mean_height() / max(a.mean_height(), 1e-6)
    if not (cfg.min_height_ratio <= h_ratio <= cfg.max_height_ratio):
        return INF
    if emb_a is None or emb_b is None:
        return INF
    return float(1.0 - np.dot(emb_a, emb_b))


def assign_links(
    tracklets: list[Tracklet],
    embeddings: dict[int, np.ndarray | None],
    cfg: StitchCfg,
) -> tuple[list[tuple[int, int]], list[dict]]:
    """Globally optimal set of (tid_a -> tid_b) links via Hungarian matching.

    Every tracklet may have at most one successor and one predecessor;
    links costing more than appearance_thresh are rejected (no-match).

    Also returns the gated candidate list (with costs) for debugging:
    inspecting these costs is how the appearance threshold gets tuned.
    """
    n = len(tracklets)
    if n < 2:
        return [], []
    cost = np.full((n, n), INF)
    candidates = []
    for i, a in enumerate(tracklets):
        for j, b in enumerate(tracklets):
            if i != j:
                c = link_cost(a, b, embeddings.get(a.tid), embeddings.get(b.tid), cfg)
                cost[i, j] = c
                if c < INF:
                    candidates.append(
                        {
                            "from_tid": a.tid, "to_tid": b.tid,
                            "cost": round(float(c), 4),
                            "gap_s": round(b.start - a.end, 1),
                            "accepted": False,
                        }
                    )
    # Pad with a no-match block so every row can opt out at threshold cost.
    padded = np.full((n, 2 * n), cfg.appearance_thresh, dtype=np.float64)
    padded[:, :n] = cost
    rows, cols = linear_sum_assignment(padded)
    links = []
    accepted = set()
    for r, c in zip(rows, cols):
        if c < n and cost[r, c] < cfg.appearance_thresh:
            links.append((tracklets[r].tid, tracklets[c].tid))
            accepted.add((tracklets[r].tid, tracklets[c].tid))
    for cand in candidates:
        cand["accepted"] = (cand["from_tid"], cand["to_tid"]) in accepted
    return links, candidates


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def stitch_tracklets(
    store: TrackletStore, cfg: StitchCfg, device: str = "cpu"
) -> tuple[dict[int, int], dict]:
    """Returns (mapping tracker_id -> person_id, debug info).

    person_ids are 1-based, ordered by first appearance. With stitching
    disabled or nothing to merge the mapping is a relabelling of tracker
    IDs. Debug info records the embedding backend and every gated
    candidate link with its cost — the raw material for threshold tuning."""
    tracklets = store.by_start_time()
    tids = [tr.tid for tr in tracklets]
    debug: dict = {"backend": None, "n_tracklets": len(tracklets), "candidates": [], "links": []}
    if cfg.enabled and len(tracklets) > 1:
        embedder = CropEmbedder(cfg.backend, cfg.reid_weights, device, cfg.allow_fallback)
        debug["backend"] = embedder.kind
        if embedder.kind != "osnet":
            log.warning(
                "STITCHING IS USING THE %s FALLBACK, NOT A RE-ID MODEL. "
                "appearance_thresh=%.2f was tuned for OSNet cosine distances "
                "and is likely meaningless here — expect both missed and "
                "spurious merges.", embedder.kind.upper(), cfg.appearance_thresh,
            )
        embeddings = {
            tr.tid: embedder.embed_crops(tr.best_crops(cfg.crops_per_track))
            for tr in tracklets
        }
        debug["tracklets_without_embedding"] = [t for t, e in embeddings.items() if e is None]
        links, candidates = assign_links(tracklets, embeddings, cfg)
        debug["candidates"] = candidates
        debug["links"] = [{"from_tid": a, "to_tid": b} for a, b in links]
        log.info("Stitching merged %d links across %d tracklets", len(links), len(tracklets))
    else:
        links = []
    uf = _UnionFind(tids)
    for a, b in links:
        uf.union(a, b)
    # Renumber roots by first appearance.
    root_first_seen: dict[int, float] = {}
    for tr in tracklets:
        root = uf.find(tr.tid)
        root_first_seen.setdefault(root, tr.start)
    ordered_roots = sorted(root_first_seen, key=root_first_seen.get)
    root_to_pid = {root: i + 1 for i, root in enumerate(ordered_roots)}
    return {tid: root_to_pid[uf.find(tid)] for tid in tids}, debug
