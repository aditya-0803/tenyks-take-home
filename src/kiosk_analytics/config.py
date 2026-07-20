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
    appearance_thresh: float = 0.45
    max_speed_px_s: float = 400.0
    min_height_ratio: float = 0.6
    max_height_ratio: float = 1.7
    crops_per_track: int = 8
    save_debug: bool = True       # dump stitch_debug.json + crop montages
    allow_fallback: bool = False  # permit non-re-ID (ImageNet) embeddings


@dataclass
class AnalyticsCfg:
    membership: str = "bottom_strip"  # bottom_strip | anchor
    anchor: str = "bottom_center"     # used when membership == "anchor"
    strip_frac: float = 0.25          # bottom fraction of the box tested
    min_overlap: float = 0.5          # strip-in-zone fraction required
    hysteresis_samples: int = 3
    merge_gap_s: float = 3.0
    min_engagement_s: float = 8.0
    min_track_len_s: float = 1.0


@dataclass
class VizCfg:
    enabled: bool = True
    show_zone: bool = True


@dataclass
class Config:
    zone_polygon: list[list[float]] = field(default_factory=list)
    video: VideoCfg = field(default_factory=VideoCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
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
    "stitch": StitchCfg,
    "analytics": AnalyticsCfg,
    "viz": VizCfg,
}
