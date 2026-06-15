#!/usr/bin/env python3
"""Build each cat's gallery / colour gallery / centroid from refs/ (spec §6.6).

Must mirror the runtime exactly: same grayscale embedding domain and same colour
threshold, and it must write gallery.npy + color_gallery.npy (not just the legacy
centroid.npy), or the live pipeline would compare grayscale visit vectors against
colour-space centroids and lose the multi-vector / colour-bonus matching.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from stupid_cat.config import load_config
from stupid_cat.reid import Embedder, build_gallery_from_refs, centroid_from_gallery

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/cats"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(
        args.config,
        local_path=Path("config.local.yaml") if Path("config.local.yaml").exists() else None,
    )
    embedder = Embedder(
        device=cfg.inference.device,
        backbone=cfg.inference.reid_backbone,
        fp16=cfg.inference.fp16,
        grayscale=cfg.inference.reid_grayscale,  # match the runtime feature space
    )

    for cat_dir in sorted(args.data.iterdir()):
        if not cat_dir.is_dir():
            continue
        refs = cat_dir / "refs"
        if not refs.is_dir():
            continue
        gallery, color_gallery = build_gallery_from_refs(
            embedder,
            refs,
            cfg.preprocess,
            cfg.cats.min_refs,
            color_min_saturation=cfg.inference.color_min_saturation,
        )
        if gallery is None:
            logger.info("%s: too few usable refs, skipped", cat_dir.name)
            continue
        np.save(cat_dir / "gallery.npy", gallery)
        np.save(cat_dir / "centroid.npy", centroid_from_gallery(gallery))
        if color_gallery is not None:
            np.save(cat_dir / "color_gallery.npy", color_gallery)
        else:
            (cat_dir / "color_gallery.npy").unlink(missing_ok=True)
        logger.info(
            "wrote %s (%d refs, %d colour)",
            cat_dir.name,
            gallery.shape[0],
            0 if color_gallery is None else color_gallery.shape[0],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
