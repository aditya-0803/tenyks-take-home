#!/usr/bin/env python3
"""Kiosk people count & dwell time analytics.

Usage:
    python run.py --video path/to/video.mp4 --config config/default.yaml \
        --out runs/exp1 [--gt labels/gt.csv] [--no-overlay]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kiosk_analytics.config import Config  # noqa: E402
from kiosk_analytics.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input .mp4")
    parser.add_argument("--config", default="config/default.yaml", help="Pipeline config YAML")
    parser.add_argument("--out", default="runs/latest", help="Output directory")
    parser.add_argument("--gt", default=None, help="Ground-truth CSV (person_id,start_s,end_s)")
    parser.add_argument("--no-overlay", action="store_true", help="Skip overlay video rendering")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    cfg = Config.load(args.config)
    run_pipeline(
        video_path=args.video,
        cfg=cfg,
        out_dir=args.out,
        gt_path=args.gt,
        overlay=False if args.no_overlay else None,
    )


if __name__ == "__main__":
    main()
