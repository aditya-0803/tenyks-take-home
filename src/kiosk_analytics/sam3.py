"""SAM 3 engine: concept-prompted detection, segmentation and tracking.

Replaces the detector+tracker pair when `engine: sam3`. The predictor is
prompted with a concept ("person") and returns, per frame, segmentation
masks with persistent track identities — mask-based tracking handles
overlapping people natively, which is exactly where box-IoU association
breaks down in crowded scenes.

The engine emits the same per-frame observations (track id, box, conf,
full-frame mask) as the detect_track path, so the tracklet store, zone
analytics, offline stitcher, chimera splitting, overlay and evaluation
are all unchanged.

Chunked processing: SAM3's tracker keeps a memory bank that grows with
(tracked objects x frames), so long videos are processed in fixed-length
chunks with fresh tracker state. Track ids are offset per chunk to stay
globally unique; the offline stitcher re-links fragments across chunk
boundaries (gap ~0 s, spatially coincident — trivially inside its gates).

API note: SAM3 shipped in ultralytics 8.3.237 (Nov 2025) and the result
schema may drift between versions. Parsing is defensive; if the schema
differs, run tools/probe_sam3.py and adjust _parse_result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import Sam3Cfg
from .video import FrameSample, VideoReader

log = logging.getLogger(__name__)


@dataclass
class Observation:
    tid: int
    box: np.ndarray        # xyxy, full-frame coords
    conf: float
    mask: np.ndarray | None  # full-frame bool mask


class Sam3Engine:
    def __init__(self, cfg: Sam3Cfg, roi: list[int] | None = None):
        self.cfg = cfg
        self.roi = roi
        try:
            from ultralytics.models.sam import SAM3VideoSemanticPredictor
        except ImportError as e:
            raise ImportError(
                "SAM3 support requires ultralytics>=8.3.237 "
                "(pip install -U ultralytics). Weights are gated: request "
                "access at huggingface.co/facebook/sam3 and place sam3.pt "
                "in the working directory."
            ) from e
        self._predictor_cls = SAM3VideoSemanticPredictor

    # ------------------------------------------------------------------
    def _new_predictor(self):
        overrides = dict(
            task="segment",
            mode="predict",
            model=self.cfg.model,
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            save=False,
            verbose=False,
        )
        if self.cfg.quantize:
            overrides["quantize"] = self.cfg.quantize
        return self._predictor_cls(overrides=overrides)

    def stream(
        self, reader: VideoReader
    ) -> Iterator[tuple[FrameSample, list[Observation]]]:
        """Yields (frame sample, observations) for every sampled frame."""
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
    def _run_chunk(self, samples: list[FrameSample], tid_offset: int, chunk_idx: int):
        log.info(
            "SAM3 chunk %d: %d frames (t=%.1fs..%.1fs), tid offset %d",
            chunk_idx, len(samples), samples[0].t, samples[-1].t, tid_offset,
        )
        rx1 = ry1 = 0
        frames = []
        for s in samples:
            img = s.image
            if self.roi is not None:
                rx1, ry1, rx2, ry2 = self.roi
                img = img[ry1:ry2, rx1:rx2]
            frames.append(img)

        predictor = self._new_predictor()  # fresh tracker state per chunk
        results = predictor(source=frames, text=[self.cfg.prompt], stream=True)

        max_tid = tid_offset - 1
        for sample, result in zip(samples, results):
            obs = self._parse_result(
                result, sample.image.shape[:2], (rx1, ry1), tid_offset
            )
            for o in obs:
                max_tid = max(max_tid, o.tid)
            yield sample, obs
        del predictor  # release memory bank before the next chunk
        return max_tid + 1

    # ------------------------------------------------------------------
    def _parse_result(
        self,
        result,
        full_hw: tuple[int, int],
        offset: tuple[int, int],
        tid_offset: int,
    ) -> list[Observation]:
        import cv2

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = (
            boxes.conf.cpu().numpy()
            if getattr(boxes, "conf", None) is not None
            else np.ones(len(xyxy), dtype=np.float32)
        )
        ids = getattr(boxes, "id", None)
        if ids is None:
            raise RuntimeError(
                "SAM3 returned no track ids — the result schema of your "
                "ultralytics version differs from the expected one. Run "
                "`python tools/probe_sam3.py --video <mp4>` and report the "
                "printed schema so the parser can be adjusted."
            )
        ids = ids.cpu().numpy().astype(int)

        mask_data = None
        if getattr(result, "masks", None) is not None:
            mask_data = result.masks.data.cpu().numpy()

        ox, oy = offset
        full_h, full_w = full_hw
        observations = []
        for i in range(len(xyxy)):
            box = xyxy[i] + np.array([ox, oy, ox, oy], dtype=np.float32)
            mask_full = None
            if mask_data is not None and i < len(mask_data):
                m = mask_data[i]
                crop_h = (self.roi[3] - self.roi[1]) if self.roi else full_h
                crop_w = (self.roi[2] - self.roi[0]) if self.roi else full_w
                if m.shape != (crop_h, crop_w):
                    m = cv2.resize(m.astype(np.float32), (crop_w, crop_h))
                mask_full = np.zeros((full_h, full_w), dtype=bool)
                mask_full[oy:oy + crop_h, ox:ox + crop_w] = m > 0.5
            observations.append(
                Observation(
                    tid=int(ids[i]) + tid_offset,
                    box=box,
                    conf=float(conf[i]),
                    mask=mask_full,
                )
            )
        return observations
