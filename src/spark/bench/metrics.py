"""Classification metrics and synthetic stress tests on patches."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def binary_metrics(y: NDArray[np.int64], p: NDArray[np.float32],
                   threshold: float = 0.5) -> dict[str, float]:
    yhat = (p >= threshold).astype(np.int64)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "n": len(y),
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": (tpr + tnr) / 2,
        "f1": 2 * prec * tpr / max(prec + tpr, 1e-12),
        "auc": roc_auc(y, p),
        "false_free_rate": fn / max(tp + fn, 1),  # occupied bays reported free
        "false_busy_rate": fp / max(tn + fp, 1),  # free bays reported occupied
    }


def roc_auc(y: NDArray[np.int64], p: NDArray[np.float32]) -> float:
    """Mann-Whitney U formulation, with ties counted as half."""
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    ranks = np.empty(len(p), np.float64)
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def occlude(patches: NDArray[np.uint8], fraction: float, seed: int = 0) -> NDArray[np.uint8]:
    """Cover ``fraction`` of each patch with one grey-ish box anchored to a random edge.

    A box hugging an edge is what a neighbouring car or a passing van looks like from a
    pole-mounted camera; a floating box in the middle is not.
    """
    if fraction <= 0:
        return patches
    rng = np.random.default_rng(seed)
    out = patches.copy()
    n, h, w = patches.shape[:3]
    for i in range(n):
        side = rng.integers(4)
        colour = rng.integers(40, 200, size=3, dtype=np.uint8)
        if side in (0, 1):  # left or right band
            bw = max(1, round(w * fraction))
            sl = (slice(None), slice(0, bw)) if side == 0 else (slice(None), slice(w - bw, w))
        else:  # top or bottom band
            bh = max(1, round(h * fraction))
            sl = (slice(0, bh), slice(None)) if side == 2 else (slice(h - bh, h), slice(None))
        out[i][sl] = colour
    return out


def darken(patches: NDArray[np.uint8], gamma: float) -> NDArray[np.uint8]:
    """gamma > 1 darkens, like dusk; applied on 0..1 intensities."""
    if gamma == 1.0:
        return patches
    lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).round().astype(np.uint8)
    return lut[patches]
