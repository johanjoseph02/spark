"""Export a trained SparkNet to ONNX, optionally quantised to int8.

The ``.onnx`` file is all the edge device needs, plus a ``.json`` sidecar with the input
size and normalisation, read by ``spark.vision.classifiers.OnnxClassifier``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from spark.vision.cnn import SparkNet, count_macs, count_params
from spark.vision.train import MEAN, STD


def _sidecar(model: SparkNet, name: str, extra: dict[str, object]) -> dict[str, object]:
    cfg = model.config
    return {
        "name": name,
        "input_size": cfg.input_size,
        "width": cfg.width,
        "mean": list(MEAN),
        "std": list(STD),
        "params": count_params(model),
        "macs": count_macs(model, cfg.input_size),
        "output": "logit",
        **extra,
    }


def export_onnx(model: SparkNet, path: str | Path, extra: dict[str, object] | None = None
                ) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = model.cpu().eval()
    s = m.config.input_size
    torch.onnx.export(
        m,
        (torch.zeros(1, 3, s, s),),
        str(path),
        input_names=["patches"],
        output_names=["logits"],
        dynamic_axes={"patches": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    path.with_suffix(".json").write_text(
        json.dumps(_sidecar(m, m.config.name, extra or {}), indent=2) + "\n")
    return path


class _Calibration:
    def __init__(self, batches: list[NDArray[np.float32]], name: str) -> None:
        self._it = iter(batches)
        self._name = name

    def get_next(self) -> dict[str, NDArray[np.float32]] | None:
        b = next(self._it, None)
        return None if b is None else {self._name: b}


def quantize_int8(fp32_path: str | Path, calib_patches: NDArray[np.uint8],
                  out_path: str | Path) -> Path:
    """Static int8 (QDQ, per-channel weights) calibrated on real patches."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from spark.vision.classifiers import OnnxClassifier

    fp32_path, out_path = Path(fp32_path), Path(out_path)
    clf = OnnxClassifier(fp32_path)
    x = clf.preprocess(calib_patches)
    batches = [x[i:i + 64] for i in range(0, len(x), 64)]

    class Reader(CalibrationDataReader):  # type: ignore[misc]
        def __init__(self) -> None:
            self.inner = _Calibration(batches, clf.input_name)

        def get_next(self) -> dict[str, NDArray[np.float32]] | None:
            return self.inner.get_next()

    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(fp32_path), str(pre))
        quantize_static(
            str(pre), str(out_path), Reader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
        )
    meta = json.loads(fp32_path.with_suffix(".json").read_text())
    meta["name"] = f"{meta['name']}-int8"
    meta["quantization"] = "static int8, QDQ, per-channel weights"
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_path
