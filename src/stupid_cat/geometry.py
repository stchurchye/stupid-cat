"""BBox vs ROI polygon overlap (spec §8.2)."""

from __future__ import annotations

import cv2
import numpy as np

BBox = tuple[float, float, float, float]
Point = tuple[float, float]
Polygon = list[Point] | list[list[float]]

_MIN_ROI_POINTS = 3


def _normalize_bbox(bbox: BBox) -> BBox:
    x1, y1, x2, y2 = bbox
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def is_convex_polygon(polygon: Polygon) -> bool:
    """True if polygon has >= 3 points and is convex (required for intersectConvexConvex)."""
    if len(polygon) < _MIN_ROI_POINTS:
        return False

    pts = [(float(p[0]), float(p[1])) for p in polygon]
    sign = 0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        x3, y3 = pts[(i + 2) % n]
        cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        if abs(cross) < 1e-9:
            continue
        current = 1 if cross > 0 else -1
        if sign == 0:
            sign = current
        elif current != sign:
            return False
    return sign != 0


def bbox_roi_overlap_ratio(bbox: BBox, roi_polygon: Polygon) -> float:
    """Return area(bbox ∩ ROI) / area(bbox). Both in the same coordinate space."""
    if len(roi_polygon) < _MIN_ROI_POINTS:
        return 0.0

    x1, y1, x2, y2 = _normalize_bbox(bbox)
    area_bbox = (x2 - x1) * (y2 - y1)
    if area_bbox <= 0:
        return 0.0

    bbox_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    roi_poly = np.array(roi_polygon, dtype=np.float32)

    _, intersect = cv2.intersectConvexConvex(bbox_poly, roi_poly)
    if intersect is None or len(intersect) == 0:
        return 0.0

    return float(cv2.contourArea(intersect) / area_bbox)
