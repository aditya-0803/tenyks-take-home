"""End-to-end pipeline: detect -> track -> stitch -> analytics -> viz/eval."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import compute_person_results, summarize
from .config import Config
from .detect import PersonDetector, resolve_device  # noqa: F401 (resolve_device used by sam3 branch)
from .eval import evaluate, load_gt
from .stitch import stitch_tracklets
from .track import Tracker
from .tracklets import TrackletStore
from .video import VideoReader
from .viz import render_overlay
from .zone import Zone

log = logging.getLogger(__name__)


def run_pipeline(
    video_path: str | Path,
    cfg: Config,
    out_dir: str | Path,
    gt_path: str | Path | None = None,
    overlay: bool | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    zone = Zone(cfg.zone_polygon)
    reader = VideoReader(video_path, cfg.video.target_fps)
    log.info(
        "Video: %dx%d, %.2f fps source, sampling at %.2f fps (stride %d)",
        reader.width, reader.height, reader.src_fps, reader.sample_fps, reader.stride,
    )

    store = TrackletStore()
    n_processed = 0

    # ---- pass 1: person observations (engine-dependent) -------------------
    if cfg.engine in ("sam3", "sam2_hybrid"):
        device = resolve_device(cfg.detector.device)
        if cfg.engine == "sam3":
            from .sam3 import Sam3Engine

            engine = Sam3Engine(cfg.sam3, roi=cfg.detector.roi)
        else:
            from .sam2_hybrid import Sam2HybridEngine

            engine = Sam2HybridEngine(cfg.sam2_hybrid, cfg.detector)
        for sample, observations in engine.stream(reader):
            boxes = np.array([o.box for o in observations]).reshape(-1, 4)
            contaminated = _contamination_flags(boxes)
            for i, o in enumerate(observations):
                store.add_observation(
                    o.tid, sample.index, sample.t, o.box, o.conf, sample.image,
                    mask=o.mask, contaminated=bool(contaminated[i]),
                )
            n_processed += 1
            if n_processed % 500 == 0:
                log.info("Processed %d frames (t=%.1fs)", n_processed, sample.t)
    elif cfg.engine == "detect_track":
        detector = PersonDetector(cfg.detector)
        device = detector.device
        tracker = Tracker(cfg.tracker, device=device)
        for sample in reader:
            dets, masks = detector(sample.image)
            contaminated = _contamination_flags(dets)
            tracks = tracker.update(dets, sample.image)
            for x1, y1, x2, y2, tid, conf, det_ind in tracks:
                di = int(det_ind)
                mask = masks[di] if masks is not None and 0 <= di < len(masks) else None
                dirty = bool(contaminated[di]) if 0 <= di < len(contaminated) else True
                store.add_observation(
                    int(tid), sample.index, sample.t,
                    np.array([x1, y1, x2, y2]), float(conf), sample.image,
                    mask=mask, contaminated=dirty,
                )
            n_processed += 1
            if n_processed % 500 == 0:
                log.info("Processed %d frames (t=%.1fs)", n_processed, sample.t)
    else:
        raise ValueError(f"Unknown engine '{cfg.engine}' (detect_track | sam3)")
    log.info("Tracking done: %d raw tracklets from %d frames", len(store), n_processed)

    # ---- pass 2: offline stitching ----------------------------------------
    tid_to_pid, stitch_debug = stitch_tracklets(store, cfg.stitch, device=device)
    if cfg.stitch.save_debug:
        _dump_stitch_debug(out_dir, store, tid_to_pid, stitch_debug)

    identities: dict[int, dict] = {}
    for tid, tr in store.tracklets.items():
        pid = tid_to_pid[tid]
        d = identities.setdefault(
            pid, {"times": [], "frame_indices": [], "boxes": [], "source_tids": []}
        )
        d["times"].extend(tr.times)
        d["frame_indices"].extend(tr.frame_indices)
        d["boxes"].extend(tr.boxes)
        d["source_tids"].append(tid)
    for d in identities.values():
        order = np.argsort(d["times"])
        d["times"] = np.asarray(d["times"])[order]
        d["frame_indices"] = np.asarray(d["frame_indices"])[order]
        d["boxes"] = np.asarray(d["boxes"])[order]

    # ---- analytics ----------------------------------------------------------
    results, timelines = compute_person_results(
        identities, zone, cfg.analytics, reader.sample_period,
        frame_size=(reader.width, reader.height),
    )
    summary = summarize(results)
    summary["runtime_s"] = round(time.time() - t0, 1)
    # Peak GPU memory: evidence for the 16 GB edge-deployment constraint.
    # (nvidia-smi shows ~0.5-1 GB more: CUDA context isn't counted here.)
    try:
        import torch

        if torch.cuda.is_available():
            summary["peak_vram_alloc_gb"] = round(
                torch.cuda.max_memory_allocated() / 2**30, 2
            )
            summary["peak_vram_reserved_gb"] = round(
                torch.cuda.max_memory_reserved() / 2**30, 2
            )
    except Exception:  # noqa: BLE001 - metrics are best-effort
        pass
    summary["raw_tracklets"] = len(store)
    summary["stitch_backend"] = stitch_debug.get("backend")
    summary["stitch_merges"] = len(stitch_debug.get("merges", []))

    results_df = pd.DataFrame([asdict(r) for r in results]).sort_values("pid")
    results_df.to_csv(out_dir / "persons.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- evaluation ---------------------------------------------------------
    if gt_path:
        metrics = evaluate(results, load_gt(gt_path))
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        summary["metrics"] = metrics

    # ---- overlay ------------------------------------------------------------
    do_overlay = cfg.viz.enabled if overlay is None else overlay
    if do_overlay:
        engaged_pids = {r.pid for r in results if r.engaged}
        render_overlay(
            video_path, out_dir / "overlay.mp4", timelines,
            engaged_pids, cfg.analytics.min_engagement_s, cfg,
        )

    log.info("Summary: %s", json.dumps(summary, indent=2, default=str))
    return summary


def _contamination_flags(dets: np.ndarray, iom_thresh: float = 0.2) -> np.ndarray:
    """Flag detections whose box materially overlaps another detection
    (intersection / own-area > iom_thresh). Crops from such frames contain
    pieces of two people (segmentation masks bleed at boundaries), so they
    are excluded from re-ID crop harvesting."""
    n = len(dets)
    flags = np.zeros(n, dtype=bool)
    if n < 2:
        return flags
    for i in range(n):
        x1, y1, x2, y2 = dets[i, :4]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0:
            flags[i] = True
            continue
        for j in range(n):
            if i == j:
                continue
            xx1, yy1 = max(x1, dets[j, 0]), max(y1, dets[j, 1])
            xx2, yy2 = min(x2, dets[j, 2]), min(y2, dets[j, 3])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            if inter / area > iom_thresh:
                flags[i] = True
                break
    return flags


def _dump_stitch_debug(out_dir: Path, store, tid_to_pid: dict, debug: dict) -> None:
    """Write stitch_debug.json plus one crop-montage image per tracklet so
    embedding quality and threshold placement can be inspected by eye."""
    import cv2

    (out_dir / "stitch_debug.json").write_text(json.dumps(debug, indent=2))
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    for tid, tr in store.tracklets.items():
        crops = tr.best_crops(6)
        if crops:
            montage = np.hstack(crops)
            pid = tid_to_pid.get(tid, 0)
            cv2.imwrite(str(crops_dir / f"pid{pid:03d}_tid{tid}.jpg"), montage)
