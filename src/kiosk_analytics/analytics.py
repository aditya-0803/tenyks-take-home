"""Dwell time and distinct-person count from stitched identities.

Dwell policy (documented in docs/LABELING.md and mirrored by the ground
truth protocol): a person's dwell time is the SUM of their in-zone
segments; time spent away from the zone between segments is excluded.
A person counts as "engaged" (and contributes to the distinct-person
count) only if their total dwell >= min_engagement_s, which filters
walk-throughs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AnalyticsCfg
from .zone import Zone, anchor_points


@dataclass
class PersonResult:
    pid: int
    dwell_s: float
    engaged: bool
    segments: list[tuple[float, float]]
    first_seen: float
    last_seen: float
    source_tids: list[int] = field(default_factory=list)


@dataclass
class PersonTimeline:
    """Per-sample state for one identity, used by the overlay renderer."""

    pid: int
    times: np.ndarray        # (N,) sample timestamps
    frame_indices: np.ndarray  # (N,)
    boxes: np.ndarray        # (N, 4)
    in_zone: np.ndarray      # (N,) smoothed state
    cum_dwell: np.ndarray    # (N,) accumulated dwell at each sample


def _in_zone_flags(
    boxes: np.ndarray,
    zone: Zone,
    cfg: AnalyticsCfg,
    zone_mask: np.ndarray | None,
) -> np.ndarray:
    """Zone membership per box.

    "bottom_strip" (default): fraction of the box's bottom strip covered by
    the zone mask must reach min_overlap. Robust to boxes clipped by the
    frame edge and to feet occluded by the kiosk itself — failure modes of
    the single-point anchor test.
    "anchor": point-in-polygon test of one anchor point.
    """
    if cfg.membership == "anchor" or zone_mask is None:
        return zone.contains(anchor_points(boxes, cfg.anchor))
    if cfg.membership != "bottom_strip":
        raise ValueError(f"Unknown membership mode '{cfg.membership}'")
    h, w = zone_mask.shape
    flags = np.zeros(len(boxes), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        sy1 = y2 - cfg.strip_frac * (y2 - y1)
        ix1, ix2 = int(max(x1, 0)), int(min(x2, w))
        iy1, iy2 = int(max(sy1, 0)), int(min(y2, h))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        flags[i] = zone_mask[iy1:iy2, ix1:ix2].mean() >= cfg.min_overlap
    return flags


def _debounce(raw: np.ndarray, k: int) -> np.ndarray:
    """Flip state only after k consecutive samples of the opposite state."""
    if k <= 1 or len(raw) == 0:
        return raw.copy()
    out = np.empty_like(raw)
    state = bool(raw[0])
    run = 0
    for i, v in enumerate(raw):
        if bool(v) != state:
            run += 1
            if run >= k:
                state = bool(v)
                run = 0
        else:
            run = 0
        out[i] = state
    return out


def _segments_from_state(times: np.ndarray, state: np.ndarray, max_dt: float) -> list[tuple[float, float]]:
    """Maximal in-zone intervals. Samples further apart than max_dt do not
    bridge a segment (the person was undetected; don't credit that time)."""
    segments: list[tuple[float, float]] = []
    start = None
    prev_t = None
    for t, s in zip(times, state):
        if s:
            if start is None or (prev_t is not None and t - prev_t > max_dt):
                if start is not None:
                    segments.append((start, prev_t))
                start = t
            prev_t = t
        else:
            if start is not None:
                segments.append((start, prev_t))
                start = None
            prev_t = t
    if start is not None:
        segments.append((start, prev_t))
    return [(s, e) for s, e in segments if e > s]


def merge_segments(segments: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    if not segments:
        return []
    segments = sorted(segments)
    merged = [list(segments[0])]
    for s, e in segments[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged]


def compute_person_results(
    identities: dict[int, dict],
    zone: Zone,
    cfg: AnalyticsCfg,
    sample_period: float,
    frame_size: tuple[int, int] | None = None,
) -> tuple[list[PersonResult], list[PersonTimeline]]:
    """identities: pid -> dict(times, frame_indices, boxes, source_tids),
    arrays sorted by time (built by pipeline from stitched tracklets).
    frame_size: (width, height); required for bottom_strip membership."""
    results: list[PersonResult] = []
    timelines: list[PersonTimeline] = []
    bridge_dt = 3.0 * sample_period  # unseen gaps longer than this don't count
    zone_mask = None
    if cfg.membership == "bottom_strip" and frame_size is not None:
        zone_mask = zone.mask(frame_size[1], frame_size[0])

    for pid, data in identities.items():
        times = np.asarray(data["times"])
        boxes = np.asarray(data["boxes"])
        if len(times) == 0 or times[-1] - times[0] < cfg.min_track_len_s:
            continue
        raw = _in_zone_flags(boxes, zone, cfg, zone_mask)
        state = _debounce(raw, cfg.hysteresis_samples)

        segments = _segments_from_state(times, state, max_dt=bridge_dt)
        segments = merge_segments(segments, cfg.merge_gap_s)
        dwell = float(sum(e - s for s, e in segments))
        engaged = dwell >= cfg.min_engagement_s

        # cumulative dwell aligned to samples, for the overlay
        cum = np.zeros(len(times))
        acc = 0.0
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            if state[i] and state[i - 1] and dt <= bridge_dt:
                acc += dt
            cum[i] = acc

        results.append(
            PersonResult(
                pid=pid,
                dwell_s=round(dwell, 2),
                engaged=engaged,
                segments=[(round(s, 2), round(e, 2)) for s, e in segments],
                first_seen=round(float(times[0]), 2),
                last_seen=round(float(times[-1]), 2),
                source_tids=list(data.get("source_tids", [])),
            )
        )
        timelines.append(
            PersonTimeline(
                pid=pid,
                times=times,
                frame_indices=np.asarray(data["frame_indices"]),
                boxes=boxes,
                in_zone=state,
                cum_dwell=cum,
            )
        )
    return results, timelines


def summarize(results: list[PersonResult]) -> dict:
    engaged = [r for r in results if r.engaged]
    return {
        "distinct_people_engaged": len(engaged),
        "tracks_analyzed": len(results),
        "mean_dwell_s": round(float(np.mean([r.dwell_s for r in engaged])), 2) if engaged else 0.0,
        "median_dwell_s": round(float(np.median([r.dwell_s for r in engaged])), 2) if engaged else 0.0,
    }
