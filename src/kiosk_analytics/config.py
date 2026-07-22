"""Typed pipeline configuration loaded from YAML."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class VideoCfg:
    target_fps: float = 10.0


@dataclass
class DetectorCfg:
    model: str = "yolo11m.pt"
    conf: float = 0.3
    iou: float = 0.7
    imgsz: int = 1280
    device: str = "auto"
    # Extra NMS applied to the detector's output. NMS-free detectors
    # (RT-DETR) can emit duplicate boxes for one person in crowds; the
    # duplicates spawn phantom tracks and cause ID churn. None = off.
    nms_iou: float | None = None
    # Drop detections mostly contained inside a higher-confidence box
    # (intersection / own-area). Duplicate partial boxes (torso inside a
    # full-body box) survive plain NMS and spawn parallel phantom tracks
    # that poison identity clustering. None = off.
    containment_iom: float | None = 0.8
    # Veto implausibly WIDE detections (width/height above this): a box
    # that swallows two adjacent people is much wider than any single
    # standing person. During the 1-2s of a crossing there is simply no
    # detection; both tracks coast and re-lock on separation, preventing
    # ID theft at the source. None = off.
    max_wh_ratio: float | None = None
    # Region of interest [x1, y1, x2, y2]: detect only inside this crop
    # (coordinates mapped back to full frame). Upscaling the crop to imgsz
    # raises effective resolution on the queue (fewer merged boxes) and
    # removes irrelevant dining-area detections. None = full frame.
    roi: list[int] | None = None


@dataclass
class TrackerCfg:
    type: str = "bytetrack"
    reid_weights: str = "osnet_x0_25_msmt17.pt"
    params: dict = field(default_factory=dict)


@dataclass
class StitchCfg:
    enabled: bool = True
    backend: str = "auto"
    reid_weights: str = "osnet_x0_25_msmt17.pt"
    max_gap_s: float = 150.0
    # Tolerate small temporal overlap between linked tracklets: at an ID
    # switch the dying track coasts a few frames while its replacement
    # starts, so the two fragments of one person briefly coexist.
    max_overlap_s: float = 1.0
    # Overlapping tracklets may only merge if they are in the same PLACE
    # during the overlap (coasting duplicate of one person). Two real
    # people whose fragments overlap briefly stand apart and stay split.
    overlap_max_dist_px: float = 120.0
    appearance_thresh: float = 0.45
    max_speed_px_s: float = 400.0
    min_height_ratio: float = 0.6
    max_height_ratio: float = 1.7
    crops_per_track: int = 8
    save_debug: bool = True       # dump stitch_debug.json + crop montages
    allow_fallback: bool = False  # permit non-re-ID (ImageNet) embeddings
    # Chimera detection: split a tracklet whose two temporal halves differ
    # in appearance by more than this cosine distance (identity theft at a
    # crossing leaves one tracklet containing two people). None = off.
    chimera_thresh: float | None = 0.35
    chimera_max_crops: int = 40   # crops embedded per tracklet for the check


@dataclass
class AnalyticsCfg:
    membership: str = "hybrid"        # hybrid | bottom_strip | anchor
    anchor: str = "bottom_center"     # used when membership == "anchor"
    strip_frac: float = 0.25          # bottom fraction of the box tested
    min_overlap: float = 0.5          # strip-in-zone fraction required
    box_overlap: float = 0.35         # hybrid: full-box-in-zone fraction that
                                      # also counts (feet occluded by kiosk)
    hysteresis_samples: int = 3
    merge_gap_s: float = 3.0
    min_engagement_s: float = 8.0
    min_track_len_s: float = 1.0


@dataclass
class Sam3Cfg:
    """SAM 3 engine (concept-prompted segment + track), ultralytics>=8.3.237.

    Replaces detector+tracker when Config.engine == "sam3". Weights are
    gated: request access at huggingface.co/facebook/sam3, download
    sam3.pt manually."""

    model: str = "sam3.pt"
    prompt: str = "person"
    conf: float = 0.25
    imgsz: int = 768
    quantize: int | None = 16   # fp16: halves VRAM, keeps 16GB budget safe
    # Process the video in chunks: SAM3's memory bank grows with tracked
    # objects x frames. A fresh tracker state per chunk bounds VRAM; the
    # offline stitcher glues chunk-boundary fragments (gap ~0s, spatially
    # coincident), so identities survive.
    chunk_s: float = 60.0


@dataclass
class Sam2HybridCfg:
    """Hybrid engine: YOLO discovers people, SAM2.1's memory tracker owns
    their identities (mask propagation, not box-IoU association). Needs
    `pip install git+https://github.com/facebookresearch/sam2`; the
    checkpoint auto-downloads (ungated)."""

    checkpoint: str = "sam2.1_hiera_small.pt"
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    chunk_s: float = 20.0        # fresh tracker state per chunk bounds VRAM;
                                 # stitcher glues chunk-boundary fragments
    prompt_conf: float = 0.5     # YOLO conf required to seed a new SAM2 object
    new_person_iou: float = 0.3  # detection is "unexplained" if IoU with every
                                 # tracked mask-box stays below this
    new_person_min_frames: int = 3  # consecutive unexplained frames to count
                                    # as a real new entrant (kills flicker)
    max_repropagations: int = 5  # discovery iterations per chunk
    min_mask_area: int = 150     # px; smaller masks = person effectively gone
    dedup_mask_iou: float = 0.6  # two objects on one person: drop the newer


@dataclass
class VizCfg:
    enabled: bool = True
    show_zone: bool = True
    show_out_of_zone: bool = False  # draw thin gray boxes for tracked
                                    # people currently outside the zone


@dataclass
class Config:
    zone_polygon: list[list[float]] = field(default_factory=list)
    engine: str = "detect_track"  # detect_track | sam3 | sam2_hybrid
    video: VideoCfg = field(default_factory=VideoCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    sam3: Sam3Cfg = field(default_factory=Sam3Cfg)
    sam2_hybrid: Sam2HybridCfg = field(default_factory=Sam2HybridCfg)
    stitch: StitchCfg = field(default_factory=StitchCfg)
    analytics: AnalyticsCfg = field(default_factory=AnalyticsCfg)
    viz: VizCfg = field(default_factory=VizCfg)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        kwargs = {}
        for f_ in dataclasses.fields(cls):
            if f_.name not in raw:
                continue
            value = raw[f_.name]
            if dataclasses.is_dataclass(f_.type) or f_.name in _SECTIONS:
                section_cls = _SECTIONS[f_.name]
                unknown = set(value) - {x.name for x in dataclasses.fields(section_cls)}
                if unknown:
                    raise ValueError(f"Unknown keys in config section '{f_.name}': {unknown}")
                kwargs[f_.name] = section_cls(**value)
            else:
                kwargs[f_.name] = value
        return cls(**kwargs)


_SECTIONS = {
    "video": VideoCfg,
    "detector": DetectorCfg,
    "tracker": TrackerCfg,
    "sam3": Sam3Cfg,
    "sam2_hybrid": Sam2HybridCfg,
    "stitch": StitchCfg,
    "analytics": AnalyticsCfg,
    "viz": VizCfg,
}
