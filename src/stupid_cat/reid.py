"""Re-ID embeddings, fusion, and centroid matching (spec §8.3)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FUSION_MODES = frozenset({"weighted_median", "weighted_mean", "best_frame"})

# Conservative reference-image quality gate (spec §6.6): reject crops that are
# too small or essentially blank (a flat patch with no texture) so they don't
# poison a cat's centroid. The gate keys on dynamic range, NOT absolute
# brightness, so a genuinely dark IR crop of a black cat (low mean but real
# texture) still passes — only uniform all-black / all-white patches are dropped.
_REF_MIN_SIDE = 16
# p99-p1 pixel spread; below this the crop is effectively flat. Uses percentiles
# (not raw max-min) so a single hot/dead sensor pixel can't make a blank patch
# look textured, and a modest threshold so genuinely low-contrast dark IR crops
# of a black cat still pass.
_REF_MIN_RANGE = 6


@dataclass
class EmbeddingRecord:
    embedding: np.ndarray
    weight: float
    bbox_area: float
    timestamp: float


class EmbeddingBuffer:
    """Bounded visit embedding buffer (spec §8.6)."""

    def __init__(self, max_frames: int) -> None:
        self.max_frames = max_frames
        self._records: list[EmbeddingRecord] = []

    def add(self, record: EmbeddingRecord) -> None:
        self._records.append(record)
        if len(self._records) <= self.max_frames:
            return
        self._records.sort(key=lambda r: (r.bbox_area, r.timestamp))
        self._records.pop(0)

    def embeddings_and_weights(self) -> tuple[list[np.ndarray], list[float]]:
        return (
            [r.embedding for r in self._records],
            [r.weight for r in self._records],
        )

    def __len__(self) -> int:
        return len(self._records)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return float(np.dot(a_n, b_n))


def _weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
    """Exact weighted median: smallest value whose cumulative weight reaches half.

    Uses true weights (no integer replication), so fractional per-camera weights
    such as 0.5 / 1.5 actually influence the result instead of all rounding to 1.
    """
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cutoff = 0.5 * float(w.sum())
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, cutoff, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def fuse_embeddings(
    embeddings: list[np.ndarray],
    weights: list[float],
    mode: str,
    *,
    centroids: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if not embeddings:
        raise ValueError("embeddings must not be empty")
    if len(embeddings) != len(weights):
        raise ValueError("embeddings and weights length mismatch")
    if mode not in FUSION_MODES:
        raise ValueError(f"unsupported fusion mode: {mode}")

    embs = [np.asarray(e, dtype=np.float32) for e in embeddings]
    dim = embs[0].shape[0]
    if any(emb.shape[0] != dim for emb in embs):
        raise ValueError("all embeddings must have the same dimension")

    if mode == "best_frame" and not centroids:
        mode = "weighted_median"

    if mode == "weighted_mean":
        w = np.asarray(weights, dtype=np.float32)
        total = w.sum()
        if total <= 0:
            w = np.ones(len(embs), dtype=np.float32) / len(embs)
        else:
            w = w / total
        merged = np.average(np.stack(embs), axis=0, weights=w)
        return l2_normalize(merged)

    if mode == "weighted_median":
        stack = np.stack(embs)  # (N, dim)
        w = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(embs), dtype=np.float64)
        merged = np.array(
            [_weighted_median_1d(stack[:, d], w) for d in range(dim)],
            dtype=np.float32,
        )
        return l2_normalize(merged)

    assert mode == "best_frame"
    if not centroids:
        raise ValueError("best_frame fusion requires centroids")

    best_emb = embs[0]
    best_score = -1.0
    for emb in embs:
        score = max(cosine_similarity(emb, c) for c in centroids.values())
        if score > best_score:
            best_score = score
            best_emb = emb
    return l2_normalize(best_emb)


def max_centroid_similarity(
    embedding: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> float:
    """Highest cosine similarity to any known centroid (spec §9.3 correction frame)."""
    if not centroids:
        return 0.0
    return max(cosine_similarity(embedding, c) for c in centroids.values())


def match_cat(
    visit_vector: np.ndarray,
    centroids: dict[str, np.ndarray],
    threshold: float,
) -> tuple[str, float]:
    if not centroids:
        return "unknown", 0.0

    best_id = "unknown"
    best_score = 0.0
    for cat_id, centroid in centroids.items():
        score = cosine_similarity(visit_vector, centroid)
        if score > best_score:
            best_score = score
            best_id = cat_id

    if best_score < threshold:
        return "unknown", best_score
    return best_id, best_score


def load_centroid(path: Path | str) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    return l2_normalize(np.load(path, allow_pickle=False))


def ref_quality_ok(frame: np.ndarray) -> bool:
    """True if a reference crop is large enough and has real texture.

    Uses dynamic range (max-min) rather than absolute brightness so a dark IR
    crop of a black cat is kept while a flat all-black/all-white patch is rejected.
    """
    if frame.ndim < 2:
        return False
    h, w = frame.shape[:2]
    if min(h, w) < _REF_MIN_SIDE:
        return False
    lo, hi = np.percentile(frame, (1, 99))
    return float(hi) - float(lo) >= _REF_MIN_RANGE


def build_centroid_from_refs(
    embedder: Embedder,
    refs_dir: Path,
    preprocess_cfg: object,
    min_refs: int,
) -> np.ndarray | None:
    """Mean L2-normalized embedding over ref images (spec §6.6).

    Low-quality crops (too small or essentially blank) are skipped so they do not
    contaminate the centroid; if fewer than ``min_refs`` usable crops remain, no
    centroid is written.
    """
    import cv2

    from stupid_cat.preprocess import preprocess_frame

    paths = sorted(refs_dir.glob("*.jpg")) + sorted(refs_dir.glob("*.jpeg")) + sorted(
        refs_dir.glob("*.png")
    )
    if len(paths) < min_refs:
        return None

    vectors: list[np.ndarray] = []
    skipped = 0
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            skipped += 1
            continue
        if not ref_quality_ok(frame):
            logger.warning("skipping low-quality reference image %s", path)
            skipped += 1
            continue
        frame = preprocess_frame(frame, preprocess_cfg)
        vectors.append(embedder.embed(frame))

    if skipped:
        logger.info("%s: used %d refs, skipped %d", refs_dir, len(vectors), skipped)
    if len(vectors) < min_refs:
        return None

    mean = np.mean(np.stack(vectors), axis=0)
    return l2_normalize(mean)


def load_all_centroids(cats_dir: Path | str) -> dict[str, np.ndarray]:
    cats_dir = Path(cats_dir)
    centroids: dict[str, np.ndarray] = {}
    if not cats_dir.exists():
        return centroids
    for cat_dir in cats_dir.iterdir():
        if not cat_dir.is_dir():
            continue
        centroid_path = cat_dir / "centroid.npy"
        centroid = load_centroid(centroid_path)
        if centroid is not None:
            centroids[cat_dir.name] = centroid
    return centroids


class Embedder:
    """Lazy-loaded EfficientNet-B0 feature extractor."""

    def __init__(
        self,
        device: str = "cpu",
        backbone: str = "efficientnet_b0",
        *,
        fp16: bool = False,
    ) -> None:
        if backbone != "efficientnet_b0":
            raise ValueError(f"unsupported backbone: {backbone}")
        self.device = device
        # FP16 only on CUDA — half precision on CPU/MPS is unsupported or slower.
        self._use_half = bool(fp16) and str(device).startswith("cuda")
        self._model = None
        self._transform = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from torchvision import models

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval()
        model.to(self.device)
        if self._use_half:
            model.half()
        self._model = model
        self._transform = weights.transforms()

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2
        import torch

        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            raise ValueError("crop_bgr must be HxWx3 BGR")
        if crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
            raise ValueError("crop is too small to embed")

        # cv2 conversion yields a contiguous RGB array (no negative-stride view
        # that PIL/torch would silently copy or reject).
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        with self._lock:
            self._ensure_loaded()
            assert self._transform is not None
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
            tensor = self._transform(tensor).unsqueeze(0).to(self.device)
            if self._use_half:
                tensor = tensor.half()
            with torch.no_grad():
                features = self._model(tensor)
            vec = features.squeeze(0).float().cpu().numpy()
            return l2_normalize(vec)
