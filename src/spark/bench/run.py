"""The full benchmark behind docs/benchmark.md.

    uv run spark bench run            # about an hour on an M1 Max
    uv run spark bench run --quick    # minutes; small subsets, for checking the pipeline

Results are written to ``results/benchmark.json`` after every step, so an interrupted
run resumes where it stopped.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spark.bench.latency import measure
from spark.bench.metrics import binary_metrics, darken, occlude
from spark.data.cnrpark import PatchSet, Split, benchmark_splits, load_cache
from spark.vision.classifiers import (
    EdgeEnergyClassifier,
    HogLinearClassifier,
    OccupancyClassifier,
    hog_features,
    load_classifier,
)

WIDTHS = (0.25, 0.5, 1.0)
SIZES = (32, 48, 64)
OCCLUSION = (0.0, 0.1, 0.2, 0.3, 0.4)
GAMMAS = (1.0, 1.5, 2.0, 3.0)


class Results:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    def has(self, *keys: str) -> bool:
        d: Any = self.data
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return False
            d = d[k]
        return True

    def put(self, value: Any, *keys: str) -> None:
        d = self.data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")


def _subsample(idx: NDArray[np.int64], n: int | None, seed: int = 0) -> NDArray[np.int64]:
    if n is None or len(idx) <= n:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))


def _evaluate(predict: Callable[[NDArray[np.uint8]], NDArray[np.float32]], ps: PatchSet,
              idx: NDArray[np.int64]) -> dict[str, Any]:
    p = predict(ps.images[idx])
    y = ps.labels[idx]
    out: dict[str, Any] = binary_metrics(y, p)
    weather = np.array([ps.samples[i].weather for i in idx])
    out["by_weather"] = {
        w: binary_metrics(y[weather == w], p[weather == w])["accuracy"]
        for w in ("SUNNY", "OVERCAST", "RAINY") if (weather == w).any()
    }
    return out


def _robustness(predict: Callable[[NDArray[np.uint8]], NDArray[np.float32]],
                images: NDArray[np.uint8], y: NDArray[np.int64]) -> dict[str, Any]:
    return {
        "occlusion": {f"{f:.1f}": binary_metrics(y, predict(occlude(images, f)))["accuracy"]
                      for f in OCCLUSION},
        "gamma": {f"{g:.1f}": binary_metrics(y, predict(darken(images, g)))["accuracy"]
                  for g in GAMMAS},
    }


def run(root: Path, results_path: Path, models_dir: Path, quick: bool = False,
        epochs: int = 8, widths: tuple[float, ...] = WIDTHS,
        sizes: tuple[int, ...] = SIZES) -> dict[str, Any]:
    import torch

    from spark.vision.cnn import SparkNetConfig
    from spark.vision.export import export_onnx, quantize_int8
    from spark.vision.train import TrainConfig, predict, train

    res = Results(results_path)
    models_dir.mkdir(parents=True, exist_ok=True)
    cap_train = 6000 if quick else None
    cap_test = 3000 if quick else None
    cap_robust = 2000 if quick else 20000
    epochs = 1 if quick else epochs
    if quick:
        widths, sizes = widths[:1], sizes[-1:]

    sets = {s: load_cache(root, s) for s in sorted(set(sizes) | {64})}
    ps64 = sets[64]
    splits: list[Split] = benchmark_splits(ps64, root)
    res.put({s.name: {"description": s.description, "train": len(s.train),
                      "val": len(s.val), "test": len(s.test)} for s in splits},
            "splits")
    res.put({"quick": quick, "epochs": epochs, "torch": torch.__version__,
             "started": time.strftime("%Y-%m-%d %H:%M")}, "run")

    for split in splits:
        tr = _subsample(split.train, cap_train)
        va = _subsample(split.val, cap_test, 1)
        te = _subsample(split.test, cap_test, 2)
        y = ps64.labels
        print(f"[{split.name}] train {len(tr):,} val {len(va):,} test {len(te):,}", flush=True)

        if not res.has("accuracy", split.name, "edge-energy") or (
                split.name == "standard" and not (models_dir / "edge-energy.npz").exists()):
            edge = EdgeEnergyClassifier.fit(ps64.images[tr], y[tr])
            res.put(_evaluate(edge.predict_proba, ps64, te), "accuracy", split.name,
                    "edge-energy")
            if split.name == "standard":
                edge.save(models_dir / "edge-energy.npz")

        if not res.has("accuracy", split.name, "hog-linear") or (
                split.name == "standard" and not (models_dir / "hog-linear.npz").exists()):
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            hog_tr = _subsample(tr, 40000, 3)
            f = hog_features(ps64.images[hog_tr])
            scaler = StandardScaler().fit(f)
            lr = LogisticRegression(C=0.05, max_iter=400).fit(scaler.transform(f), y[hog_tr])
            hog = HogLinearClassifier(lr.coef_.ravel().astype(np.float32),
                                      float(lr.intercept_[0]),
                                      scaler.mean_.astype(np.float32),
                                      scaler.scale_.astype(np.float32))
            res.put(_evaluate(hog.predict_proba, ps64, te), "accuracy", split.name,
                    "hog-linear")
            if split.name == "standard":
                hog.save(models_dir / "hog-linear.npz")

        for size in sizes:
            ps = sets[size]
            for width in widths:
                net = SparkNetConfig(width=width, input_size=size)
                exported = (models_dir / f"{net.name}-int8.onnx").exists()
                if res.has("accuracy", split.name, net.name) and (
                        split.name != "standard" or exported):
                    continue
                model, info = train(net, ps.images, ps.labels, tr, va,
                                    TrainConfig(epochs=epochs))
                entry = _evaluate(lambda x, m=model: predict(m, x), ps, te)
                entry.update(info)
                res.put(entry, "accuracy", split.name, net.name)
                if split.name == "standard":
                    path = export_onnx(model, models_dir / f"{net.name}.onnx",
                                       {"trained_on": "CNRPark+EXT train days",
                                        "val_acc": info["val_acc"]})
                    calib = ps.images[_subsample(tr, 512, 4)]
                    quantize_int8(path, calib, models_dir / f"{net.name}-int8.onnx")
                torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    # ---------------------------------------------------------------- deployed artefacts
    std = next(s for s in splits if s.name == "standard")
    te = _subsample(std.test, cap_test, 2)
    rob = _subsample(std.test, cap_robust, 5)
    artefacts = sorted(models_dir.glob("*.onnx")) + sorted(models_dir.glob("*.npz"))
    for path in artefacts:
        clf: OccupancyClassifier = load_classifier(path)
        ps = sets.get(clf.input_size, ps64)
        key = path.stem
        if not res.has("deployed", key, "accuracy"):
            res.put(_evaluate(clf.predict_proba, ps, te), "deployed", key, "accuracy")
        if not res.has("deployed", key, "robustness"):
            res.put(_robustness(clf.predict_proba, ps.images[rob], ps.labels[rob]),
                    "deployed", key, "robustness")
        if not res.has("deployed", key, "latency"):
            res.put([measure(clf, b, repeats=10 if quick else 30) for b in (1, 32)],
                    "deployed", key, "latency")
        meta_path = path.with_suffix(".json")
        info: dict[str, Any] = {"bytes": path.stat().st_size}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            info.update({k: meta[k] for k in ("params", "macs", "input_size", "width")
                         if k in meta})
        elif isinstance(clf, HogLinearClassifier):
            info.update({"params": clf.n_params, "input_size": clf.input_size})
        else:
            info.update({"params": 2, "input_size": clf.input_size})
        res.put(info, "deployed", key, "size")

    res.put(time.strftime("%Y-%m-%d %H:%M"), "run", "finished")
    return res.data


def config_dict() -> dict[str, Any]:
    from spark.vision.train import TrainConfig

    return {"widths": WIDTHS, "sizes": SIZES, "train": asdict(TrainConfig())}
