"""Occupancy classifiers that run without a deep-learning framework.

Every classifier maps a batch of BGR uint8 patches to P(occupied). The deployed runtime
needs only numpy, OpenCV and onnxruntime: training-time libraries (torch, scikit-learn)
write their parameters into ``.npz`` or ``.onnx`` files that this module reads.

* ``EdgeEnergyClassifier``: mean absolute Laplacian of the grey patch, squashed through a
  fitted logistic curve. This is the hand-tuned method of the 2024 prototype, kept as the
  historical baseline.
* ``HogLinearClassifier``: HOG descriptor and a linear model.
* ``OnnxClassifier``: any exported CNN (see ``spark.vision.export``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

Patches = NDArray[np.uint8]
Probs = NDArray[np.float32]


@runtime_checkable
class OccupancyClassifier(Protocol):
    name: str
    input_size: int

    def predict_proba(self, patches: Patches) -> Probs: ...


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    out: NDArray[np.float64] = 0.5 * (1.0 + np.tanh(0.5 * z))
    return out


def _resize_batch(patches: Patches, size: int) -> Patches:
    if patches.shape[1] == size and patches.shape[2] == size:
        return patches
    return np.stack(
        [cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA) for p in patches]
    ).astype(np.uint8)


def _grey(patches: Patches) -> NDArray[np.uint8]:
    if patches.ndim == 3:
        return patches
    return np.stack([cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) for p in patches]).astype(np.uint8)


# --------------------------------------------------------------------------- edge energy


def edge_energy(patches: Patches, size: int = 64) -> NDArray[np.float64]:
    g = _grey(_resize_batch(patches, size)).astype(np.float32)
    out = np.empty(len(g), np.float64)
    for i, p in enumerate(g):
        p = cv2.GaussianBlur(p, (3, 3), 0)
        out[i] = float(np.abs(cv2.Laplacian(p, cv2.CV_32F, ksize=3)).mean())
    return out


class EdgeEnergyClassifier:
    """P(occupied) = sigmoid(a * log(edge energy) + b). Cars are busier than asphalt."""

    def __init__(self, a: float, b: float, input_size: int = 64) -> None:
        self.name = "edge-energy"
        self.a, self.b, self.input_size = a, b, input_size

    def predict_proba(self, patches: Patches) -> Probs:
        e = np.log(edge_energy(patches, self.input_size) + 1e-3)
        return _sigmoid(self.a * e + self.b).astype(np.float32)

    @staticmethod
    def fit(patches: Patches, labels: NDArray[np.int_], input_size: int = 64,
            iters: int = 500) -> EdgeEnergyClassifier:
        """One-feature logistic regression by Newton's method (no scikit-learn needed)."""
        x = np.log(edge_energy(patches, input_size) + 1e-3)
        y = labels.astype(np.float64)
        mu, sd = x.mean(), x.std() + 1e-9
        z = (x - mu) / sd
        w = np.zeros(2)
        for _ in range(iters):
            p = _sigmoid(w[0] * z + w[1])
            g = np.array([np.dot(p - y, z), np.sum(p - y)])
            s = p * (1 - p) + 1e-9
            h = np.array([[np.dot(s, z * z), np.dot(s, z)], [np.dot(s, z), s.sum()]])
            h += 1e-6 * np.eye(2)
            step = np.linalg.solve(h, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        return EdgeEnergyClassifier(w[0] / sd, w[1] - w[0] * mu / sd, input_size)

    def save(self, path: str | Path) -> None:
        np.savez(path, kind="edge-energy", a=self.a, b=self.b, input_size=self.input_size)


# --------------------------------------------------------------------------- HOG + linear

_HOG_SIZE = 64


def _hog() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor((_HOG_SIZE, _HOG_SIZE), (16, 16), (8, 8), (8, 8), 9)


def hog_features(patches: Patches) -> NDArray[np.float32]:
    hog = _hog()
    g = _grey(_resize_batch(patches, _HOG_SIZE))
    return np.stack([np.asarray(hog.compute(p)).ravel() for p in g]).astype(np.float32)


class HogLinearClassifier:
    def __init__(self, coef: NDArray[np.float32], intercept: float,
                 mean: NDArray[np.float32], scale: NDArray[np.float32]) -> None:
        self.name = "hog-linear"
        self.input_size = _HOG_SIZE
        self.coef, self.intercept = coef.astype(np.float32), float(intercept)
        self.mean, self.scale = mean.astype(np.float32), scale.astype(np.float32)

    def predict_proba(self, patches: Patches) -> Probs:
        f = (hog_features(patches) - self.mean) / self.scale
        return _sigmoid((f @ self.coef + self.intercept).astype(np.float64)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        np.savez(path, kind="hog-linear", coef=self.coef, intercept=self.intercept,
                 mean=self.mean, scale=self.scale)

    @property
    def n_params(self) -> int:
        return int(self.coef.size + 1)


# --------------------------------------------------------------------------- ONNX


class OnnxClassifier:
    """A CNN exported by ``spark.vision.export``. Metadata lives in a ``.json`` sidecar."""

    def __init__(self, path: str | Path, threads: int = 1) -> None:
        import onnxruntime as ort

        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        self.name = str(meta["name"])
        self.input_size = int(meta["input_size"])
        self.mean = np.array(meta["mean"], np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(meta["std"], np.float32).reshape(1, 3, 1, 1)
        self.meta = meta
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, patches: Patches) -> NDArray[np.float32]:
        x = _resize_batch(patches, self.input_size).astype(np.float32) / 255.0
        x = x[..., ::-1].transpose(0, 3, 1, 2)  # BGR -> RGB, NHWC -> NCHW
        return np.ascontiguousarray((x - self.mean) / self.std, dtype=np.float32)

    def predict_proba(self, patches: Patches) -> Probs:
        if len(patches) == 0:
            return np.zeros(0, np.float32)
        (logits,) = self.session.run(None, {self.input_name: self.preprocess(patches)})
        z = np.asarray(logits, np.float64).reshape(len(patches), -1)
        p = _sigmoid(z[:, 0]) if z.shape[1] == 1 else _sigmoid(z[:, 1] - z[:, 0])
        return p.astype(np.float32)


# --------------------------------------------------------------------------- loading


def load_classifier(path: str | Path, threads: int = 1) -> OccupancyClassifier:
    path = Path(path)
    if path.suffix == ".onnx":
        return OnnxClassifier(path, threads=threads)
    if path.suffix == ".npz":
        d = np.load(path)
        kind = str(d["kind"])
        if kind == "edge-energy":
            return EdgeEnergyClassifier(float(d["a"]), float(d["b"]), int(d["input_size"]))
        if kind == "hog-linear":
            return HogLinearClassifier(d["coef"], float(d["intercept"]), d["mean"], d["scale"])
        raise ValueError(f"unknown classifier kind {kind!r} in {path}")
    raise ValueError(f"don't know how to load {path} (expected .onnx or .npz)")
