import pytest

from stupid_cat.geometry import bbox_roi_overlap_ratio, is_convex_polygon


def test_bbox_roi_overlap_ratio() -> None:
    bbox = (100, 100, 200, 200)
    roi = [(0, 0), (300, 0), (300, 300), (0, 300)]
    r = bbox_roi_overlap_ratio(bbox, roi)
    assert 0.0 < r <= 1.0


def test_bbox_fully_inside_roi() -> None:
    bbox = (50, 50, 100, 100)
    roi = [(0, 0), (200, 0), (200, 200), (0, 200)]
    assert bbox_roi_overlap_ratio(bbox, roi) == 1.0


def test_bbox_disjoint_from_roi() -> None:
    bbox = (400, 400, 500, 500)
    roi = [(0, 0), (200, 0), (200, 200), (0, 200)]
    assert bbox_roi_overlap_ratio(bbox, roi) == 0.0


def test_bbox_partial_overlap_exact_ratio() -> None:
    bbox = (150, 150, 250, 250)
    roi = [(0, 0), (200, 0), (200, 200), (0, 200)]
    assert bbox_roi_overlap_ratio(bbox, roi) == pytest.approx(0.25)


def test_bbox_normalizes_inverted_coordinates() -> None:
    bbox = (200, 200, 100, 100)
    roi = [(0, 0), (300, 0), (300, 300), (0, 300)]
    assert bbox_roi_overlap_ratio(bbox, roi) == bbox_roi_overlap_ratio((100, 100, 200, 200), roi)


def test_is_convex_polygon() -> None:
    assert is_convex_polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert not is_convex_polygon([(0, 0), (100, 0), (0, 100), (100, 100)])


def test_empty_roi_returns_zero() -> None:
    assert bbox_roi_overlap_ratio((10, 10, 20, 20), []) == 0.0
