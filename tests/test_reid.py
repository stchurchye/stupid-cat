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
