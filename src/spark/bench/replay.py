"""End-to-end check of the deployed pipeline on whole camera frames.

The patch benchmark scores classifiers on the dataset's own 150x150 crops. A deployed
camera does something different: it cuts bays out of a full 1000x750 frame using the
lot geometry, then counts free spaces. This module runs exactly that path
(``Lot`` -> ``extract_patches`` -> classifier) on the test days of CNRPark+EXT and
scores what a driver would see: the free-space count per frame.

    uv run spark bench replay --model models/default.onnx
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from spark.data.cnrpark import read_labels
from spark.lot import lot_from_cnrpark_csv
from spark.vision.classifiers import load_classifier
from spark.vision.patches import extract_patches


def _frame_key(path: Path) -> tuple[str, str]:
    """``2015-11-12_0709.jpg`` -> ("2015-11-12", "07.09")."""
    date, hm = path.stem.split("_")
    return date, f"{hm[:2]}.{hm[2:]}"


def replay(root: Path, model_path: Path, cameras: tuple[int, ...] = tuple(range(1, 10))
           ) -> dict[str, Any]:
    raw = root / "raw"
    clf = load_classifier(model_path)
    test_days = set((raw / "splits" / "CNRPark-EXT" / "test_days.txt").read_text().split())
    truth: dict[tuple[int, str, str], dict[str, int]] = defaultdict(dict)
    for s in read_labels(raw / "cnr-ext" / "LABELS" / "all.txt"):
        if s.date in test_days:
            truth[(s.camera, s.date, s.time)][s.slot] = s.label

    per_camera: dict[str, Any] = {}
    all_err: list[int] = []
    bay_hits = bay_total = 0
    for cam in cameras:
        lot = lot_from_cnrpark_csv(raw / "full" / f"camera{cam}.csv", f"cnr-{cam}", f"Camera {cam}")
        frames = sorted((raw / "full" / "FULL_IMAGE_1000x750").glob(f"*/*/camera{cam}/*.jpg"))
        errs: list[int] = []
        hits = total = 0
        for path in frames:
            date, hm = _frame_key(path)
            labels = truth.get((cam, date, hm))
            if not labels:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            ids, patches = extract_patches(np.asarray(img, np.uint8), lot, clf.input_size)
            p = clf.predict_proba(patches)
            pred = {i: int(q >= 0.5) for i, q in zip(ids, p, strict=True)}
            common = [i for i in ids if i in labels]
            if not common:
                continue
            hits += sum(pred[i] == labels[i] for i in common)
            total += len(common)
            free_true = sum(1 - labels[i] for i in common)
            free_pred = sum(1 - pred[i] for i in common)
            errs.append(abs(free_pred - free_true))
        if not errs:
            continue
        e = np.array(errs)
        per_camera[str(cam)] = {
            "frames": len(errs),
            "bays_scored": total,
            "bay_accuracy": hits / total,
            "free_count_mae": float(e.mean()),
            "frames_exact": float((e == 0).mean()),
            "frames_within_1": float((e <= 1).mean()),
            "frames_within_2": float((e <= 2).mean()),
        }
        all_err += errs
        bay_hits += hits
        bay_total += total
    e = np.array(all_err)
    return {
        "model": clf.name,
        "test_days": sorted(test_days),
        "overall": {
            "frames": len(all_err),
            "bays_scored": bay_total,
            "bay_accuracy": bay_hits / max(bay_total, 1),
            "free_count_mae": float(e.mean()) if len(e) else None,
            "frames_exact": float((e == 0).mean()) if len(e) else None,
            "frames_within_1": float((e <= 1).mean()) if len(e) else None,
            "frames_within_2": float((e <= 2).mean()) if len(e) else None,
        },
        "per_camera": per_camera,
    }


def main(root: Path, models: list[Path], out: Path) -> dict[str, Any]:
    """Replay each model and merge the results into ``out``, keyed by artefact name."""
    all_res: dict[str, Any] = json.loads(out.read_text()) if out.exists() else {}
    for model in models:
        all_res[model.stem] = replay(root, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_res, indent=2, sort_keys=True) + "\n")
    return all_res


def annotate(root: Path, model_path: Path, camera: int, out: Path, frame: str = "") -> Path:
    """Draw the model's verdict on one test-day frame: green free, red occupied, and a
    yellow outline wherever the model disagrees with the label."""
    raw = root / "raw"
    clf = load_classifier(model_path)
    test_days = set((raw / "splits" / "CNRPark-EXT" / "test_days.txt").read_text().split())
    all_frames = (raw / "full" / "FULL_IMAGE_1000x750").glob(f"*/*/camera{camera}/*.jpg")
    frames = sorted(p for p in all_frames if p.parent.parent.name in test_days)
    path = next((p for p in frames if p.stem == frame), None) if frame else frames[len(frames) // 2]
    if path is None:
        raise FileNotFoundError(f"no test-day frame {frame!r} for camera {camera}")
    labels = {s.slot: s.label for s in read_labels(raw / "cnr-ext" / "LABELS" / "all.txt")
              if s.camera == camera and (s.date, s.time) == _frame_key(path)}
    lot = lot_from_cnrpark_csv(raw / "full" / f"camera{camera}.csv", "x", "x").for_frame(1000, 750)
    img = np.asarray(cv2.imread(str(path), cv2.IMREAD_COLOR), np.uint8)
    ids, patches = extract_patches(img, lot, clf.input_size)
    p = clf.predict_proba(patches)
    overlay = img.copy()
    for bay, q in zip(lot.bays, p, strict=True):
        pts = np.array(bay.polygon, np.int32)
        busy = q >= 0.5
        cv2.fillPoly(overlay, [pts], (60, 60, 220) if busy else (80, 190, 80))
    out_img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for bay, q in zip(lot.bays, p, strict=True):
        pts = np.array(bay.polygon, np.int32)
        wrong = bay.id in labels and int(q >= 0.5) != labels[bay.id]
        cv2.polylines(out_img, [pts], True, (0, 220, 255) if wrong else (255, 255, 255),
                      3 if wrong else 1, cv2.LINE_AA)
    free = int((p < 0.5).sum())
    caption = f"CNRPark+EXT camera {camera}, {path.stem}: {free} of {len(ids)} free ({clf.name})"
    cv2.rectangle(out_img, (0, 0), (1000, 34), (20, 20, 20), -1)
    cv2.putText(out_img, caption, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), out_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out
