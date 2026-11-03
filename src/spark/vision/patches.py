"""Cut one fixed-size patch per bay out of a frame.

A bay drawn as an arbitrary quadrilateral is warped to a square, so the classifier always
sees the space the same way up regardless of camera angle. Other polygons fall back to
their bounding box with everything outside the polygon set to the patch mean.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from spark.lot import Bay, Lot

Image = NDArray[np.uint8]


def _order_quad(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    """Top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def extract_patch(frame: Image, bay: Bay, size: int) -> Image:
    h, w = frame.shape[:2]
    poly = np.array(bay.polygon, dtype=np.float32)
    if len(poly) == 4:
        src = _order_quad(poly)
        dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
        m = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(frame, m, (size, size), flags=cv2.INTER_AREA,
                                  borderMode=cv2.BORDER_REPLICATE)
        return np.asarray(out, dtype=np.uint8)

    x0, y0, x1, y1 = bay.bounds()
    xa, ya = max(int(np.floor(x0)), 0), max(int(np.floor(y0)), 0)
    xb, yb = min(int(np.ceil(x1)), w), min(int(np.ceil(y1)), h)
    if xb <= xa or yb <= ya:
        raise ValueError(f"bay {bay.id} lies outside the {w}x{h} frame")
    crop = frame[ya:yb, xa:xb].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(poly - [xa, ya]).astype(np.int32)], 255)
    mean = cv2.mean(crop, mask=mask)[: crop.shape[2] if crop.ndim == 3 else 1]
    crop[mask == 0] = np.array(mean, dtype=np.float64).astype(np.uint8)
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return np.asarray(out, dtype=np.uint8)


def extract_patches(frame: Image, lot: Lot, size: int) -> tuple[list[str], Image]:
    """All bays of ``lot`` from ``frame`` as an ``(n, size, size, 3)`` uint8 array."""
    h, w = frame.shape[:2]
    lot = lot.for_frame(w, h)
    ids = [b.id for b in lot.bays]
    batch = np.stack([extract_patch(frame, b, size) for b in lot.bays]) if ids else np.zeros(
        (0, size, size, 3), np.uint8
    )
    return ids, batch
