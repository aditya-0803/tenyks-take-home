"""Hybrid engine: YOLO discovery + SAM 2.1 memory tracking.

Motivation (measured, not assumed): on this footage YOLO detection is
strong, but box-IoU association (ByteTrack/BoT-SORT) breaks down whenever
>4 people share the kiosk area — IDs churn faster than the offline
stitcher can repair. SAM 2.1's video predictor tracks *masks* with a
temporal memory bank, which is robust exactly where box association is
weakest: overlap, occlusion, re-emergence. It cannot discover people by
itself (that needs a concept detector), so:

    YOLO finds each person once  ->  their box seeds a SAM2 object  ->
    SAM2's memory tracker owns that identity for the rest of the chunk.

Discovery of mid-chunk entrants: after propagation, any detection that no
tracked mask explains for several consecutive frames is a new person;
their earliest box is prompted in and the chunk is re-propagated (bounded
iterations).

Chunked processing bounds VRAM (fresh tracker state per chunk); track ids
are globally unique via per-chunk offsets, and the offline stitcher glues
chunk-boundary fragments (gap ~0 s, spatially coincident).

Install: pip install git+https://github.com/facebookresearch/sam2
Checkpoint (ungated) auto-downloads on first run.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import DetectorCfg, Sam2HybridCfg
from .detect import PersonDetector
from .sam3 import Observation
from .video import FrameSample, VideoReader

log = logging.getLogger(__name__)

_CHECKPOINT_URLS = {
    "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}


def _mask_to_box(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def find_new_entrants(
    dets_per_frame: list[np.ndarray],
    tracked_boxes_per_frame: list[list[np.ndarray]],
    already_prompted: list[tuple[int, np.ndarray]],
    iou_thresh: float,
    min_frames: int,
) -> tuple[int, list[np.ndarray]] | None:
    """Earliest (frame_idx, det boxes) of detections that no tracked mask
    explains for >= min_frames consecutive frames. Pure logic, unit-tested.

    dets_per_frame: per frame, (N, 4) xyxy detection boxes.
    tracked_boxes_per_frame: per frame, list of mask-derived boxes.
    already_prompted: (frame_idx, box) prompts from earlier iterations —
    detections matching one nearby are skipped (SAM2 may legitimately fail
    on them; re-prompting forever would loop).
    """
    n = len(dets_per_frame)
    unexplained: list[list[np.ndarray]] = []
    for i in range(n):
        row = []
        for det in dets_per_frame[i]:
            if any(_iou(det, tb) >= iou_thresh for tb in tracked_boxes_per_frame[i]):
                continue
            if any(
                abs(i - pf) <= 3 * min_frames and _iou(det, pb) >= 0.5
                for pf, pb in already_prompted
            ):
                continue
            row.append(det)
        unexplained.append(row)

    for i in range(n - min_frames + 1):
        for det in unexplained[i]:
            persistent = all(
                any(_iou(det, d2) >= 0.3 for d2 in unexplained[i + k])
                for k in range(1, min_frames)
            )
            if persistent:
                boxes = [det] + [
                    d for d in unexplained[i][1:] if not np.array_equal(d, det)
                ]
                return i, boxes
    return None


class Sam2HybridEngine:
    def __init__(self, cfg: Sam2HybridCfg, det_cfg: DetectorCfg):
        self.cfg = cfg
        self.roi = det_cfg.roi
        self.detector = PersonDetector(det_cfg)  # returns full-frame coords
        self.device = self.detector.device
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as e:
            raise ImportError(
                "The sam2_hybrid engine needs Meta's SAM2 package: "
                "pip install git+https://github.com/facebookresearch/sam2"
            ) from e
        ckpt = self._ensure_checkpoint(cfg.checkpoint)
        self.predictor = build_sam2_video_predictor(cfg.model_cfg, ckpt, device=self.device)
        log.info("SAM2 video predictor ready (%s on %s)", cfg.checkpoint, self.device)

    @staticmethod
    def _ensure_checkpoint(name: str) -> str:
        path = Path(name)
        if path.exists():
            return str(path)
        url = _CHECKPOINT_URLS.get(path.name)
        if url is None:
            raise FileNotFoundError(
                f"Checkpoint {name} not found and no known download URL. "
                f"Known: {sorted(_CHECKPOINT_URLS)}"
            )
        log.info("Downloading %s ...", url)
        urllib.request.urlretrieve(url, path)
        return str(path)

    # ------------------------------------------------------------------
    def stream(
        self, reader: VideoReader
    ) -> Iterator[tuple[FrameSample, list[Observation]]]:
        chunk_len = max(1, int(round(self.cfg.chunk_s * reader.sample_fps)))
        buffer: list[FrameSample] = []
        tid_offset = 0
        chunk_idx = 0
        for sample in reader:
            buffer.append(sample)
            if len(buffer) >= chunk_len:
                tid_offset = yield from self._run_chunk(buffer, tid_offset, chunk_idx)
                buffer = []
                chunk_idx += 1
        if buffer:
            yield from self._run_chunk(buffer, tid_offset, chunk_idx)

    # ------------------------------------------------------------------
    def _run_chunk(
        self, samples: list[FrameSample], tid_offset: int, chunk_idx: int
    ):
        log.info(
            "SAM2 chunk %d: %d frames (t=%.1fs..%.1fs)",
            chunk_idx, len(samples), samples[0].t, samples[-1].t,
        )
        rx1, ry1 = (self.roi[0], self.roi[1]) if self.roi else (0, 0)

        # YOLO discovery boxes (full-frame coords -> ROI-local for SAM2)
        dets_roi: list[np.ndarray] = []
        for s in samples:
            dets, _ = self.detector(s.image)
            strong = dets[dets[:, 4] >= self.cfg.prompt_conf][:, :4]
            strong = strong - np.array([rx1, ry1, rx1, ry1], dtype=np.float32)
            dets_roi.append(strong.astype(np.float32))

        tmpdir = Path(tempfile.mkdtemp(prefix="sam2_chunk_"))
        try:
            for i, s in enumerate(samples):
                img = s.image
                if self.roi:
                    img = img[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                cv2.imwrite(str(tmpdir / f"{i:05d}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])

            import contextlib

            import torch

            # SAM2's reference implementation runs under bf16 autocast and
            # parts of the model assume it: without this, the memory bank
            # holds bf16 tensors while later passes feed fp32 (dtype crash
            # on re-propagation after adding a prompt).
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if "cuda" in str(self.device)
                else contextlib.nullcontext()
            )
            with torch.inference_mode(), autocast:
                state = self.predictor.init_state(
                    video_path=str(tmpdir), offload_video_to_cpu=True
                )
                prompted: list[tuple[int, np.ndarray]] = []
                next_obj = 0
                for det in dets_roi[0]:
                    self.predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=next_obj, box=det
                    )
                    prompted.append((0, det))
                    next_obj += 1

                masks_by_frame: dict[int, dict[int, np.ndarray]] = {}
                for _ in range(self.cfg.max_repropagations):
                    masks_by_frame = self._propagate(state)
                    tracked_boxes = [
                        [b for b in (
                            _mask_to_box(m) for m in masks_by_frame.get(i, {}).values()
                        ) if b is not None]
                        for i in range(len(samples))
                    ]
                    entrant = find_new_entrants(
                        dets_roi, tracked_boxes, prompted,
                        self.cfg.new_person_iou, self.cfg.new_person_min_frames,
                    )
                    if entrant is None:
                        break
                    frame_idx, boxes = entrant
                    log.info(
                        "Chunk %d: %d new entrant(s) at frame %d — re-propagating",
                        chunk_idx, len(boxes), frame_idx,
                    )
                    for box in boxes:
                        self.predictor.add_new_points_or_box(
                            state, frame_idx=frame_idx, obj_id=next_obj, box=box
                        )
                        prompted.append((frame_idx, box))
                        next_obj += 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        max_tid = tid_offset - 1
        full_h, full_w = samples[0].image.shape[:2]
        for i, sample in enumerate(samples):
            obs = self._to_observations(
                masks_by_frame.get(i, {}), tid_offset, (full_h, full_w), (rx1, ry1)
            )
            for o in obs:
                max_tid = max(max_tid, o.tid)
            yield sample, obs
        return max_tid + 1

    # ------------------------------------------------------------------
    def _propagate(self, state) -> dict[int, dict[int, np.ndarray]]:
        masks_by_frame: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
            frame_masks = {}
            for j, obj_id in enumerate(obj_ids):
                mask = (mask_logits[j] > 0.0).squeeze().cpu().numpy()
                if mask.sum() >= self.cfg.min_mask_area:
                    frame_masks[int(obj_id)] = mask
            masks_by_frame[int(frame_idx)] = frame_masks
        return masks_by_frame

    def _to_observations(
        self,
        frame_masks: dict[int, np.ndarray],
        tid_offset: int,
        full_hw: tuple[int, int],
        offset: tuple[int, int],
    ) -> list[Observation]:
        ox, oy = offset
        full_h, full_w = full_hw
        entries = []
        for obj_id, mask in sorted(frame_masks.items()):
            box = _mask_to_box(mask)
            if box is None:
                continue
            entries.append((obj_id, box, mask))
        # Dedup: two SAM2 objects riding one person -> keep the older id.
        kept = []
        for obj_id, box, mask in entries:
            if any(_iou(box, kb) > self.cfg.dedup_mask_iou for _, kb, _ in kept):
                continue
            kept.append((obj_id, box, mask))

        observations = []
        for obj_id, box, mask in kept:
            mask_full = np.zeros((full_h, full_w), dtype=bool)
            mh, mw = mask.shape
            mask_full[oy:oy + mh, ox:ox + mw] = mask
            observations.append(
                Observation(
                    tid=tid_offset + obj_id,
                    box=box + np.array([ox, oy, ox, oy], dtype=np.float32),
                    conf=0.9,  # SAM2 has no per-frame confidence; fixed
                    mask=mask_full,
                )
            )
        return observations
