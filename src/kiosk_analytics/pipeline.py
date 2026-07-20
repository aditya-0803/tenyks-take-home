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
from .detect import PersonDetector, resolve_device
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

    detector = PersonDetector(cfg.detector)
    tracker = Tracker(cfg.tracker, device=detector.device)
    store = TrackletStore()

    # ---- pass 1: detect + track -------------------------------------------
    n_processed = 0
    for sample in reader:
        dets, masks = detector(sample.image)
        tracks = tracker.update(dets, sample.image)
        for x1, y1, x2, y2, tid, conf, det_ind in tracks:
            di = int(det_ind)
            mask = masks[di] if masks is not None and 0 <= di < len(masks) else None
            store.add_observation(
                int(tid), sample.index, sample.t,
                np.array([x1, y1, x2, y2]), float(conf), sample.image, mask=mask,
            )
        n_processed += 1
        if n_processed % 500 == 0:
            log.info("Processed %d frames (t=%.1fs)", n_processed, sample.t)
    log.info("Tracking done: %d raw tracklets from %d frames", len(store), n_processed)

    # ---- pass 2: offline stitching ----------------------------------------
    tid_to_pid, stitch_debug = stitch_tracklets(store, cfg.stitch, device=detector.device)
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
