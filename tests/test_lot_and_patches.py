from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from spark.lot import Bay, Lot, lot_from_cnrpark_csv
from spark.vision.patches import extract_patch, extract_patches


def test_lot_round_trips_through_json(tmp_path: Path, lot: Lot) -> None:
    path = tmp_path / "lot.json"
    lot.save(path)
    again = Lot.load(path)
    assert again == lot
    assert again.capacity == 4
    assert again.zones() == ["a", "b"]


def test_lot_rejects_duplicate_bays() -> None:
    sq = ((0, 0), (1, 0), (1, 1))
    with pytest.raises(ValueError, match="unique"):
        Lot("x", "x", (10, 10), (Bay("1", sq), Bay("1", sq)))


def test_bay_needs_three_points() -> None:
    with pytest.raises(ValueError):
        Bay("1", ((0, 0), (1, 1)))


def test_for_frame_rescales_polygons(lot: Lot) -> None:
    half = lot.for_frame(100, 50)
    assert half.frame_size == (100, 50)
    assert half.bays[0].polygon[0] == (5.0, 20.0)
    assert lot.for_frame(200, 100) is lot


def test_cnrpark_csv_import(tmp_path: Path) -> None:
    p = tmp_path / "camera1.csv"
    p.write_text("SlotId,X,Y,W,H\n601,100,200,50,60\n602,0,0,10,10\n")
    lot = lot_from_cnrpark_csv(p, "c1", "Camera 1")
    assert [b.id for b in lot.bays] == ["601", "602"]
    assert lot.bays[0].bounds() == (100, 200, 150, 260)
    scaled = lot.for_frame(1000, 750)
    assert scaled.bays[0].bounds()[0] == pytest.approx(100 * 1000 / 2592)


def test_quad_is_warped_upright() -> None:
    frame = np.zeros((100, 100, 3), np.uint8)
    # A skewed quad, red on its top half and blue on its bottom half.
    quad = np.array([[20, 10], [80, 20], [90, 90], [10, 80]], np.int32)
    cv2.fillPoly(frame, [quad], (0, 0, 255))
    cv2.fillPoly(frame, [np.array([[15, 45], [85, 55], [90, 90], [10, 80]], np.int32)],
                 (255, 0, 0))
    # Points given out of order on purpose: the extractor orders them itself.
    bay = Bay("q", ((90, 90), (20, 10), (10, 80), (80, 20)))
    patch = extract_patch(frame, bay, 32)
    assert patch.shape == (32, 32, 3)
    assert patch[4, 16, 2] > 200 and patch[4, 16, 0] < 50  # top is red (BGR)
    assert patch[28, 16, 0] > 200  # bottom is blue


def test_polygon_fallback_masks_outside() -> None:
    frame = np.full((50, 50, 3), 255, np.uint8)
    tri = Bay("t", ((10, 10), (40, 10), (25, 40)))
    patch = extract_patch(frame, tri, 16)
    assert patch.shape == (16, 16, 3)
    assert (patch == 255).all()  # outside filled with the mean of the inside


def test_bay_outside_frame_raises() -> None:
    frame = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(ValueError, match="outside"):
        extract_patch(frame, Bay("x", ((20, 20), (30, 20), (25, 30))), 8)


def test_extract_patches_rescales_to_frame(lot: Lot) -> None:
    frame = np.zeros((50, 100, 3), np.uint8)  # half the mapped size
    ids, batch = extract_patches(frame, lot, 24)
    assert ids == ["B0", "B1", "B2", "B3"]
    assert batch.shape == (4, 24, 24, 3)
