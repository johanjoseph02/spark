from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from conftest import flat, textured

from spark.bench.metrics import binary_metrics, darken, occlude, roc_auc
from spark.vision.classifiers import (
    EdgeEnergyClassifier,
    HogLinearClassifier,
    hog_features,
    load_classifier,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _data() -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([textured(60), flat(60)])
    y = np.array([1] * 60 + [0] * 60)
    return x, y


def test_edge_energy_learns_busy_vs_flat(tmp_path: Path) -> None:
    x, y = _data()
    clf = EdgeEnergyClassifier.fit(x, y)
    p = clf.predict_proba(x)
    assert ((p >= 0.5) == y).mean() > 0.95
    clf.save(tmp_path / "e.npz")
    again = load_classifier(tmp_path / "e.npz")
    assert np.allclose(again.predict_proba(x), p)


def test_hog_linear_round_trip(tmp_path: Path) -> None:
    x, _ = _data()
    f = hog_features(x)
    assert f.shape == (120, 1764)
    rng = np.random.default_rng(0)
    clf = HogLinearClassifier(rng.standard_normal(1764).astype(np.float32), 0.1,
                              f.mean(0), f.std(0) + 1e-6)
    clf.save(tmp_path / "h.npz")
    again = load_classifier(tmp_path / "h.npz")
    assert np.allclose(again.predict_proba(x), clf.predict_proba(x), atol=1e-6)


def test_unknown_artefacts_are_rejected(tmp_path: Path) -> None:
    np.savez(tmp_path / "x.npz", kind="nope")
    with pytest.raises(ValueError, match="unknown"):
        load_classifier(tmp_path / "x.npz")
    with pytest.raises(ValueError):
        load_classifier(tmp_path / "model.pt")


def test_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9], np.float32)) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1], np.float32)) == 0.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5], np.float32)) == 0.5
    m = binary_metrics(y, np.array([0.1, 0.7, 0.8, 0.4], np.float32))
    assert m["accuracy"] == 0.5
    assert m["false_free_rate"] == 0.5 and m["false_busy_rate"] == 0.5


def test_stress_transforms() -> None:
    x = np.full((3, 20, 20, 3), 128, np.uint8)
    o = occlude(x, 0.25, seed=1)
    covered = (o != 128).any(axis=3).mean(axis=(1, 2))
    assert np.allclose(covered, 0.25, atol=0.06)
    assert (darken(x, 2.0) < 128).all()
    assert darken(x, 1.0) is x


@pytest.mark.train
@pytest.mark.skipif(not HAS_TORCH, reason="needs the train extra")
def test_sparknet_export_matches_torch(tmp_path: Path) -> None:
    import torch

    from spark.vision.cnn import SparkNet, SparkNetConfig, count_macs, count_params
    from spark.vision.export import export_onnx, quantize_int8
    from spark.vision.train import to_tensor

    torch.manual_seed(0)
    net = SparkNet(SparkNetConfig(width=0.25, input_size=32)).eval()
    assert count_params(net) > 1000
    assert count_macs(net, 32) < count_macs(SparkNet(SparkNetConfig(0.5, 32)), 32)
    path = export_onnx(net, tmp_path / "m.onnx")
    x, _ = _data()
    onnx_p = load_classifier(path).predict_proba(x)
    with torch.no_grad():
        small = np.stack([__import__("cv2").resize(p, (32, 32), interpolation=3) for p in x])
        torch_p = torch.sigmoid(net(to_tensor(small, torch.device("cpu")))).squeeze(1).numpy()
    assert np.allclose(onnx_p, torch_p, atol=1e-4)

    q = quantize_int8(path, x[:32], tmp_path / "m-int8.onnx")
    qp = load_classifier(q).predict_proba(x)
    assert qp.shape == (120,)
    assert np.abs(qp - onnx_p).mean() < 0.1


@pytest.mark.train
@pytest.mark.skipif(not HAS_TORCH, reason="needs the train extra")
def test_augment_keeps_shape_and_range() -> None:
    import torch

    from spark.vision.train import augment

    x = torch.rand(16, 3, 32, 32)
    out = augment(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1
    assert not torch.allclose(out, x)
