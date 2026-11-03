from __future__ import annotations

import numpy as np
import pytest

from spark.lot import Bay, Lot


def square(x: float, y: float, s: float = 20) -> tuple[tuple[float, float], ...]:
    return ((x, y), (x + s, y), (x + s, y + s), (x, y + s))


@pytest.fixture
def lot() -> Lot:
    return Lot(
        "test-lot",
        "Test lot",
        (200, 100),
        tuple(Bay(f"B{i}", square(10 + 40 * i, 40), zone="a" if i < 2 else "b")
              for i in range(4)),
        lat=43.72,
        lon=10.42,
        meta={"currency": "EUR"},
    )


def textured(n: int, size: int = 64, seed: int = 0) -> np.ndarray:
    """Patches with lots of edges: a stand-in for a parked car."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (n, size // 8, size // 8, 3), dtype=np.uint8)
    return np.repeat(np.repeat(base, 8, axis=1), 8, axis=2)


def flat(n: int, size: int = 64, seed: int = 1) -> np.ndarray:
    """Nearly uniform grey patches: a stand-in for empty asphalt."""
    rng = np.random.default_rng(seed)
    grey = rng.integers(90, 130, (n, 1, 1, 1), dtype=np.uint8)
    noise = rng.integers(0, 3, (n, size, size, 3), dtype=np.uint8)
    return (np.broadcast_to(grey, (n, size, size, 3)) + noise).astype(np.uint8)
