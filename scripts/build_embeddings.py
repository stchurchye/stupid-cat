#!/usr/bin/env python3
"""Build per-cat centroid.npy from refs/ (spec §6.6)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from stupid_cat.config import load_config
from stupid_cat.reid import Embedder, build_centroid_from_refs

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/cats"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config, local_path=Path("config.local.yaml") if Path("config.local.yaml").exists() else None)
    embedder = Embedder(
        device=cfg.inference.device,
        backbone=cfg.inference.reid_backbone,
        fp16=cfg.inference.fp16,
    )

    for cat_dir in sorted(args.data.iterdir()):
        if not cat_dir.is_dir():
            continue
        refs = cat_dir / "refs"
        if not refs.is_dir():
            continue
        centroid = build_centroid_from_refs(embedder, refs, cfg.preprocess, cfg.cats.min_refs)
        if centroid is None:
            continue
        out = cat_dir / "centroid.npy"
        np.save(out, centroid)
        logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
