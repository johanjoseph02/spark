"""Wall-clock latency of a classifier on this machine's CPU."""

from __future__ import annotations

import statistics
import time

import numpy as np

from spark.vision.classifiers import OccupancyClassifier


def measure(clf: OccupancyClassifier, batch: int, repeats: int = 30, warmup: int = 5,
            seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    s = clf.input_size
    x = rng.integers(0, 256, size=(batch, s, s, 3), dtype=np.uint8)
    for _ in range(warmup):
        clf.predict_proba(x)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        clf.predict_proba(x)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {
        "batch": batch,
        "median_ms": med * 1e3,
        "p90_ms": float(np.percentile(times, 90)) * 1e3,
        "per_bay_ms": med * 1e3 / batch,
    }
