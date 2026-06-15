import numpy as np
import pytest

from stupid_cat.reid import fuse_embeddings, match_cat


def test_weighted_median_fusion() -> None:
    embs = [np.ones(8), np.ones(8) * 2]
    weights = [0.5, 1.0]
    v = fuse_embeddings(embs, weights, mode="weighted_median")
    assert v.shape == (8,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)


def test_weighted_mean_fusion() -> None:
    embs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    v = fuse_embeddings(embs, [1.0, 1.0], mode="weighted_mean")
    assert v.shape == (2,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)


def test_best_frame_without_centroids_falls_back_to_median() -> None:
    embs = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    v = fuse_embeddings(embs, [0.5, 1.0], mode="best_frame", centroids=None)
    expected = fuse_embeddings(embs, [0.5, 1.0], mode="weighted_median")
    assert np.allclose(v, expected)


def test_fuse_embeddings_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        fuse_embeddings([np.ones(4), np.ones(8)], [1.0, 1.0], mode="weighted_mean")


def test_load_centroid_l2_normalizes(tmp_path) -> None:
    from stupid_cat.reid import load_centroid

    path = tmp_path / "mimi.npy"
    np.save(path, np.array([3.0, 4.0], dtype=np.float32))
    centroid = load_centroid(path)
    assert centroid is not None
    assert np.isclose(np.linalg.norm(centroid), 1.0, atol=1e-6)


def test_best_frame_fusion_picks_highest_similarity() -> None:
    embs = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]
    centroids = {"mimi": np.array([0.0, 1.0])}
    v = fuse_embeddings(embs, [1.0, 1.0], mode="best_frame", centroids=centroids)
    assert np.allclose(v, np.array([0.0, 1.0]))


def test_match_cat_returns_unknown_below_threshold() -> None:
    centroids = {"mimi": np.array([1.0, 0.0])}
    cat_id, confidence = match_cat(np.array([0.0, 1.0]), centroids, threshold=0.55)
    assert cat_id == "unknown"
    assert confidence < 0.55


def test_match_cat_returns_best_above_threshold() -> None:
    centroids = {
        "mimi": np.array([1.0, 0.0]),
        "cat2": np.array([0.0, 1.0]),
    }
    visit = np.array([0.95, 0.05])
    visit = visit / np.linalg.norm(visit)
    cat_id, confidence = match_cat(visit, centroids, threshold=0.55)
    assert cat_id == "mimi"
    assert confidence >= 0.55


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
def test_embedder_produces_unit_vector() -> None:
    from stupid_cat.reid import Embedder

    embedder = Embedder(device="cpu")
    crop = np.zeros((128, 128, 3), dtype=np.uint8)
    crop[40:88, 40:88] = 180
    vec = embedder.embed(crop)
    assert vec.ndim == 1
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
def test_embedder_grayscale_embed_runs() -> None:
    from stupid_cat.reid import Embedder

    embedder = Embedder(device="cpu", grayscale=True)
    crop = np.zeros((128, 128, 3), dtype=np.uint8)
    crop[40:88, 40:88] = (20, 120, 200)  # a colour crop the grayscale path must collapse
    vec = embedder.embed(crop)
    assert vec.ndim == 1
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


# --- Day/night identity upgrade: galleries, colour bonus, backbones -----------

import cv2  # noqa: E402

from stupid_cat.config import PreprocessConfig  # noqa: E402
from stupid_cat.reid import (  # noqa: E402
    Embedder,
    build_gallery_from_refs,
    centroid_from_gallery,
    color_histogram,
    color_similarity,
    crop_is_colorful,
    match_identity,
)


class _FixedEmbedder:
    """Returns a deterministic vector per crop mean so galleries differ a little."""

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        vec = np.zeros(1280, dtype=np.float32)
        vec[0] = 1.0
        vec[1] = float(crop_bgr.mean()) / 255.0
        return vec


def _write_noise_jpg(path, *, color: bool, seed: int) -> None:
    rng = np.random.default_rng(seed)
    if color:
        img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    else:
        g = rng.integers(0, 255, (32, 32), dtype=np.uint8)
        img = cv2.merge([g, g, g])
    cv2.imwrite(str(path), img)


def test_crop_is_colorful_and_histogram_grayscale_is_none() -> None:
    gray = cv2.merge([np.full((20, 20), 120, np.uint8)] * 3)
    assert crop_is_colorful(gray) is False
    assert color_histogram(gray) is None


def test_color_histogram_normalized_and_self_similarity() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
    hist = color_histogram(img)
    assert hist is not None
    assert np.isclose(hist.sum(), 1.0, atol=1e-5)
    assert color_similarity(hist, hist) == pytest.approx(1.0)
    assert color_similarity(hist, None) == 0.0


def test_centroid_from_gallery_is_unit_norm() -> None:
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    centroid = centroid_from_gallery(gallery)
    assert np.isclose(np.linalg.norm(centroid), 1.0, atol=1e-6)


def test_match_identity_topk_gallery_picks_closest_cat() -> None:
    visit = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    galleries = {
        "mimi": np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.95, 0.05, 0.0]], dtype=np.float32),
        "cat2": np.array([[0.0, 1.0, 0.0], [0.0, 0.9, 0.1]], dtype=np.float32),
    }
    cat_id, score = match_identity(visit, galleries, {}, threshold=0.5, topk=2)
    assert cat_id == "mimi"
    assert score > 0.5


def test_match_identity_centroid_fallback_skips_dim_mismatch() -> None:
    visit = np.array([1.0, 0.0], dtype=np.float32)
    centroids = {
        "mimi": np.array([1.0, 0.0], dtype=np.float32),
        "stale": np.array([1.0, 0.0, 0.0], dtype=np.float32),  # wrong dim -> skipped
    }
    cat_id, _ = match_identity(visit, {}, centroids, threshold=0.5)
    assert cat_id == "mimi"


def test_match_identity_colour_breaks_embedding_tie_daytime_only() -> None:
    visit = np.array([1.0, 0.0], dtype=np.float32)
    galleries = {
        "a": np.array([[1.0, 0.0]], dtype=np.float32),
        "b": np.array([[1.0, 0.0]], dtype=np.float32),  # identical embeddings
    }
    red = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    blue = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    color_galleries = {"a": red.reshape(1, -1), "b": blue.reshape(1, -1)}
    cat_id, _ = match_identity(
        visit, galleries, {}, threshold=0.1, topk=1,
        visit_color=blue, color_galleries=color_galleries, color_weight=0.5,
    )
    assert cat_id == "b"  # colour disambiguates by day
    # At night (visit_color is None) colour is ignored: pure-embedding score.
    _, night_score = match_identity(
        visit, galleries, {}, threshold=0.1, topk=1,
        visit_color=None, color_galleries=color_galleries, color_weight=0.5,
    )
    assert night_score == pytest.approx(1.0)


def test_match_identity_colour_never_penalises_embedding_winner() -> None:
    # Regression: a cat WITH a (mismatching) colour gallery must not lose to a cat
    # WITHOUT one when its embedding is clearly stronger.
    visit = np.array([1.0, 0.0], dtype=np.float32)
    galleries = {
        "a": np.array([[1.0, 0.0]], dtype=np.float32),  # emb 1.00 (best)
        "b": np.array([[0.97, 0.24]], dtype=np.float32),  # emb ~0.97, no colour
    }
    visit_color = np.array([0.0, 1.0], dtype=np.float32)
    color_galleries = {"a": np.array([[1.0, 0.0]], dtype=np.float32)}  # disjoint from visit
    cat_id, _ = match_identity(
        visit, galleries, {}, threshold=0.5, topk=1,
        visit_color=visit_color, color_galleries=color_galleries, color_weight=0.3,
    )
    assert cat_id == "a"


def test_match_identity_colour_mismatch_keeps_match_above_threshold() -> None:
    # Regression: a true daytime match whose colour happens to mismatch must not be
    # demoted to 'unknown' — the embedding score alone is the threshold gate.
    visit = np.array([1.0, 0.0], dtype=np.float32)
    galleries = {"a": np.array([[0.8, 0.6]], dtype=np.float32)}  # emb 0.80
    visit_color = np.array([0.0, 1.0], dtype=np.float32)
    color_galleries = {"a": np.array([[1.0, 0.0]], dtype=np.float32)}  # mismatch
    cat_id, score = match_identity(
        visit, galleries, {}, threshold=0.7, topk=1,
        visit_color=visit_color, color_galleries=color_galleries, color_weight=0.5,
    )
    assert cat_id == "a"
    assert score >= 0.7


def test_match_identity_empty_returns_unknown() -> None:
    cat_id, score = match_identity(np.array([1.0, 0.0]), {}, {}, threshold=0.5)
    assert cat_id == "unknown"
    assert score == 0.0


def test_build_gallery_from_refs_separates_colour(tmp_path) -> None:
    refs = tmp_path / "refs"
    refs.mkdir()
    for i in range(3):
        _write_noise_jpg(refs / f"v{i}_cam1.jpg", color=True, seed=i)
    for i in range(2):
        _write_noise_jpg(refs / f"g{i}_cam1.jpg", color=False, seed=100 + i)
    gallery, color_gallery = build_gallery_from_refs(
        _FixedEmbedder(), refs, PreprocessConfig(), min_refs=5
    )
    assert gallery is not None and gallery.shape == (5, 1280)
    assert color_gallery is not None and color_gallery.shape[0] == 3  # only colourful refs
    # Below min_refs -> nothing built.
    g2, c2 = build_gallery_from_refs(_FixedEmbedder(), refs, PreprocessConfig(), min_refs=99)
    assert g2 is None and c2 is None


def test_embedder_rejects_unknown_backbone() -> None:
    with pytest.raises(ValueError, match="backbone"):
        Embedder(backbone="resnet50")


def test_embedder_accepts_dinov2_and_grayscale_flag() -> None:
    embedder = Embedder(backbone="dinov2_vits14", grayscale=True)
    assert embedder.backbone == "dinov2_vits14"
    assert embedder.grayscale is True


def test_load_identities_roundtrip(tmp_path) -> None:
    from stupid_cat.reid import l2_normalize, load_identities

    cat = tmp_path / "mimi"
    cat.mkdir()
    np.save(cat / "centroid.npy", l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32)))
    np.save(cat / "gallery.npy", np.ones((4, 3), dtype=np.float32))
    np.save(cat / "color_gallery.npy", np.ones((2, 960), dtype=np.float32))
    centroids, galleries, colors = load_identities(tmp_path)
    assert centroids["mimi"].shape == (3,)
    assert galleries["mimi"].shape == (4, 3)
    assert colors["mimi"].shape == (2, 960)
