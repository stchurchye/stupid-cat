"""Re-ID embeddings, fusion, and centroid matching (spec §8.3)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stupid_cat.config import PreprocessConfig

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


# --- Colour features (daytime-only bonus; see spec §7.2 day/night) ----------
# At night the camera uses IR illumination, so frames are effectively grayscale
# and carry no usable colour. We therefore treat colour as an *optional* signal:
# a hue/saturation histogram that is only computed when a crop is actually
# colourful, and only blended into matching when present.
_COLOR_MIN_SATURATION = 25.0  # mean HSV S below this => grayscale/IR, no colour
_COLOR_H_BINS = 30
_COLOR_S_BINS = 32


def crop_is_colorful(crop_bgr: np.ndarray, *, min_saturation: float = _COLOR_MIN_SATURATION) -> bool:
    """True if the crop has enough colour to be useful (i.e. not a night IR frame)."""
    import cv2

    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        return False
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) >= min_saturation


def color_histogram(
    crop_bgr: np.ndarray, *, min_saturation: float = _COLOR_MIN_SATURATION
) -> np.ndarray | None:
    """L1-normalized hue-saturation histogram, or None if the crop is ~grayscale.

    Returning None for IR/night crops lets callers transparently skip the colour
    bonus after dark. The L1 normalization makes histogram-intersection
    similarity land in [0, 1], comparable in scale to cosine."""
    import cv2

    if not crop_is_colorful(crop_bgr, min_saturation=min_saturation):
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [_COLOR_H_BINS, _COLOR_S_BINS], [0, 180, 0, 256])
    total = float(hist.sum())
    if total <= 0:
        return None
    return (hist.flatten() / total).astype(np.float32)


def color_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Histogram intersection of two L1-normalized colour hists, in [0, 1]."""
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(np.minimum(a, b).sum())


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
        w: np.ndarray = np.asarray(weights, dtype=np.float32)
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

    Uses the p99-p1 percentile spread (not raw max-min, which a single hot/dead
    pixel could inflate) rather than absolute brightness, so a dark IR crop of a
    black cat is kept while a flat all-black/all-white patch is rejected.
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
    preprocess_cfg: PreprocessConfig,
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


def centroid_from_gallery(gallery: np.ndarray) -> np.ndarray:
    """Mean L2-normalized vector of a gallery (the legacy single-centroid form)."""
    return l2_normalize(np.mean(np.asarray(gallery, dtype=np.float32), axis=0))


def build_gallery_from_refs(
    embedder: Embedder,
    refs_dir: Path,
    preprocess_cfg: PreprocessConfig,
    min_refs: int,
    *,
    color_min_saturation: float = _COLOR_MIN_SATURATION,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build a cat's multi-vector gallery and colour gallery from its refs.

    Unlike a single mean centroid, the embedding gallery keeps EVERY usable ref's
    vector, so top-k matching stays robust when a cat looks very different across
    poses (curled vs walking vs back-only). The colour gallery holds histograms of
    only the refs that are actually colourful (daytime); night/IR refs add none.

    Returns ``(gallery (K, dim), color_gallery (Kc, hist_dim) | None)``, or
    ``(None, None)`` if fewer than ``min_refs`` usable refs exist (so a half-built
    cat never starts matching)."""
    import cv2

    from stupid_cat.preprocess import preprocess_frame

    paths = (
        sorted(refs_dir.glob("*.jpg"))
        + sorted(refs_dir.glob("*.jpeg"))
        + sorted(refs_dir.glob("*.png"))
    )
    if len(paths) < min_refs:
        return None, None

    embeddings: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    skipped = 0
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None or not ref_quality_ok(frame):
            skipped += 1
            continue
        processed = preprocess_frame(frame, preprocess_cfg)
        embeddings.append(embedder.embed(processed))
        # Colour from the ref file as-saved (refs are stored already preprocessed,
        # so this matches the single-CLAHE colour of the live visit crop). Yields
        # None for grayscale/IR refs, which then contribute no colour gallery row.
        chist = color_histogram(frame, min_saturation=color_min_saturation)
        if chist is not None:
            colors.append(chist)

    if skipped:
        logger.info("%s: used %d refs, skipped %d", refs_dir, len(embeddings), skipped)
    if len(embeddings) < min_refs:
        return None, None

    gallery = np.stack(embeddings).astype(np.float32)
    color_gallery = np.stack(colors).astype(np.float32) if colors else None
    return gallery, color_gallery


def _topk_mean_similarity(visit_vector: np.ndarray, gallery: np.ndarray, topk: int) -> float:
    """Mean of the top-k cosine similarities between the visit and a cat's gallery.

    Top-k (not max) so a single near-duplicate ref can't dominate, and not the
    mean over all refs so an oddly-posed ref can't drag a true match down."""
    v = l2_normalize(visit_vector)
    g = np.asarray(gallery, dtype=np.float32)
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    g = g / np.maximum(norms, 1e-12)
    sims = g @ v
    k = max(1, min(int(topk), sims.shape[0]))
    return float(np.mean(np.sort(sims)[-k:]))


def match_identity(
    visit_vector: np.ndarray,
    galleries: dict[str, np.ndarray] | None,
    centroids: dict[str, np.ndarray] | None,
    threshold: float,
    *,
    topk: int = 3,
    visit_color: np.ndarray | None = None,
    color_galleries: dict[str, np.ndarray] | None = None,
    color_weight: float = 0.0,
) -> tuple[str, float]:
    """Identify a visit: embedding decides, daytime colour only breaks ties.

    For each candidate cat the embedding score is the mean of the top-k cosine
    similarities to its gallery, falling back to cosine-to-centroid when no gallery
    exists. The EMBEDDING score is the gate — a cat must clear ``threshold`` on
    embedding alone, so the rule is identical day and night and a colour mismatch
    can never push a true match below threshold. Colour is then a non-negative
    bonus (``color_weight`` × best histogram intersection, daytime only) used purely
    to RE-RANK cats that already qualify, so a cat is never penalised merely for
    having a colour gallery. Galleries/centroids whose dimension differs from the
    visit vector are skipped (e.g. after a backbone switch, until refs rebuild).

    Returns ``(cat_id, embedding_score)``; the score is the winner's calibrated
    cosine (not the colour-inflated value)."""
    galleries = galleries or {}
    centroids = centroids or {}
    color_galleries = color_galleries or {}
    cat_ids = set(galleries) | set(centroids)
    if not cat_ids:
        return "unknown", 0.0

    dim = int(np.asarray(visit_vector).shape[0])
    scored: list[tuple[str, float, float]] = []  # (cat_id, emb_score, colour_bonus)
    best_emb_seen = 0.0
    for cat_id in cat_ids:
        gallery = galleries.get(cat_id)
        if gallery is not None and gallery.ndim == 2 and gallery.shape[1] == dim:
            emb_score = _topk_mean_similarity(visit_vector, gallery, topk)
        else:
            centroid = centroids.get(cat_id)
            if centroid is None or centroid.ndim != 1 or centroid.shape[0] != dim:
                continue  # dim mismatch / no usable embedding -> skip
            emb_score = cosine_similarity(visit_vector, centroid)
        best_emb_seen = max(best_emb_seen, emb_score)

        bonus = 0.0
        if visit_color is not None and color_weight > 0.0:
            cg = color_galleries.get(cat_id)
            if cg is not None and cg.ndim == 2 and cg.shape[1] == visit_color.shape[0]:
                bonus = color_weight * max(color_similarity(visit_color, row) for row in cg)
        scored.append((cat_id, emb_score, bonus))

    qualified = [(cid, emb, bonus) for cid, emb, bonus in scored if emb >= threshold]
    if not qualified:
        return "unknown", best_emb_seen
    best_id, best_emb, _ = max(qualified, key=lambda t: t[1] + t[2])
    return best_id, best_emb


def _load_npy_2d(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        arr = np.load(path, allow_pickle=False)
    except (OSError, ValueError):  # noqa: BLE001 - corrupt file shouldn't crash startup
        logger.warning("could not load %s; ignoring", path)
        return None
    if arr.ndim != 2:
        return None
    return arr.astype(np.float32)


def load_identities(
    cats_dir: Path | str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load (centroids, galleries, color_galleries) for every cat directory.

    Centroids remain for backward compatibility / fusion; galleries drive matching
    when present; colour galleries enable the daytime bonus. Any missing file just
    leaves that cat absent from the corresponding dict."""
    cats_dir = Path(cats_dir)
    centroids: dict[str, np.ndarray] = {}
    galleries: dict[str, np.ndarray] = {}
    color_galleries: dict[str, np.ndarray] = {}
    if not cats_dir.exists():
        return centroids, galleries, color_galleries
    for cat_dir in cats_dir.iterdir():
        if not cat_dir.is_dir():
            continue
        cid = cat_dir.name
        centroid = load_centroid(cat_dir / "centroid.npy")
        if centroid is not None:
            centroids[cid] = centroid
        gallery = _load_npy_2d(cat_dir / "gallery.npy")
        if gallery is not None:
            galleries[cid] = gallery
        color_gallery = _load_npy_2d(cat_dir / "color_gallery.npy")
        if color_gallery is not None:
            color_galleries[cid] = color_gallery
    return centroids, galleries, color_galleries


REID_BACKBONES = frozenset({"efficientnet_b0", "dinov2_vits14", "dinov2_vitb14"})


class Embedder:
    """Lazy-loaded feature extractor (EfficientNet-B0 or DINOv2)."""

    def __init__(
        self,
        device: str = "cpu",
        backbone: str = "efficientnet_b0",
        *,
        fp16: bool = False,
        grayscale: bool = False,
    ) -> None:
        if backbone not in REID_BACKBONES:
            raise ValueError(
                f"unsupported backbone: {backbone} (supported: {sorted(REID_BACKBONES)})"
            )
        self.device = device
        self.backbone = backbone
        # Embed on luminance so daytime-colour and night-IR frames share one
        # domain — a single gallery then matches a cat day OR night (spec §7.2).
        self.grayscale = bool(grayscale)
        # FP16 only on CUDA — half precision on CPU/MPS is unsupported or slower.
        self._use_half = bool(fp16) and str(device).startswith("cuda")
        self._model = None
        self._transform = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch

        if self.backbone == "efficientnet_b0":
            from torchvision import models

            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
            model.classifier = torch.nn.Identity()
            self._transform = weights.transforms()
        else:  # dinov2_* — self-supervised ViT, far stronger at instance/individual ID
            from torchvision import transforms as T

            model = torch.hub.load("facebookresearch/dinov2", self.backbone)
            self._transform = T.Compose(
                [
                    T.Resize(224, antialias=True),
                    T.CenterCrop(224),
                    T.ConvertImageDtype(torch.float32),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        model.eval()
        model.to(self.device)
        if self._use_half:
            model.half()
        self._model = model

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2
        import torch

        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            raise ValueError("crop_bgr must be HxWx3 BGR")
        if crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
            raise ValueError("crop is too small to embed")

        working = crop_bgr
        if self.grayscale:
            # Collapse to luminance (replicated to 3ch) so day-colour and night-IR
            # crops embed into the same space; colour is recovered separately as a
            # daytime-only bonus (see color_histogram).
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            working = cv2.merge([gray, gray, gray])

        # cv2 conversion yields a contiguous RGB array (no negative-stride view
        # that PIL/torch would silently copy or reject).
        rgb = cv2.cvtColor(working, cv2.COLOR_BGR2RGB)

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
