"""CNRPark+EXT: 145k labelled parking-space patches from 9 cameras, Nov 2015 to Feb 2016.

Amato et al., "Deep learning for decentralized parking lot occupancy detection", Expert
Systems with Applications 72 (2017). Licensed under the Open Data Commons Open Database
License (ODbL) v1.0. http://cnrpark.it

``spark data download`` fetches the archives into ``data/raw``; ``spark data cache``
decodes every patch once into uint8 arrays at the sizes the benchmark uses.
"""

from __future__ import annotations

import csv
import shutil
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

RELEASE = "https://github.com/fabiocarrara/deep-parking/releases/download/archive"
ARCHIVES = {
    "patches": "CNR-EXT-Patches-150x150.zip",
    "splits": "splits.zip",
    "full": "CNR-EXT_FULL_IMAGE_1000x750.tar",
}
WEATHERS = ("SUNNY", "OVERCAST", "RAINY")
CACHE_SIZES = (32, 48, 64)


@dataclass(frozen=True)
class Sample:
    path: str  # relative to PATCHES/
    label: int  # 1 = occupied
    weather: str
    date: str
    camera: int
    time: str
    slot: str


def parse_path(rel: str, label: int) -> Sample:
    """``SUNNY/2015-11-12/camera8/S_2015-11-12_12.26_C08_280.jpg`` -> Sample."""
    weather, date, cam_dir, fname = rel.split("/")
    stem = fname.rsplit(".", 1)[0]
    _, _, time, _, slot = stem.split("_")
    return Sample(rel, label, weather, date, int(cam_dir.removeprefix("camera")), time, slot)


def read_labels(path: Path) -> list[Sample]:
    out: list[Sample] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rel, lab = line.rsplit(" ", 1)
            out.append(parse_path(rel, int(lab)))
    return out


# --------------------------------------------------------------------------- download


def download(root: Path, which: Iterable[str] = ("patches", "splits"),
             progress: Callable[[str], None] = print) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for key in which:
        name = ARCHIVES[key]
        dest = raw / name
        if not dest.exists():
            progress(f"downloading {name}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(f"{RELEASE}/{name}") as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh, length=1 << 20)
            tmp.rename(dest)
        progress(f"extracting {name}")
        if key == "patches":
            with zipfile.ZipFile(dest) as z:
                z.extractall(raw / "cnr-ext")
        elif key == "splits":
            with zipfile.ZipFile(dest) as z:
                z.extractall(raw)
        else:
            with tarfile.open(dest) as t:
                t.extractall(raw / "full", filter="data")


# --------------------------------------------------------------------------- cache


@dataclass
class PatchSet:
    """All CNRPark+EXT patches at one size, with their metadata, held in memory."""

    images: NDArray[np.uint8]  # (n, s, s, 3) BGR
    samples: list[Sample]
    index: dict[str, int]

    @property
    def labels(self) -> NDArray[np.int64]:
        return np.array([s.label for s in self.samples], np.int64)

    def select(self, keep: Callable[[Sample], bool]) -> NDArray[np.int64]:
        return np.array([i for i, s in enumerate(self.samples) if keep(s)], np.int64)

    def rows(self, samples: Iterable[Sample]) -> NDArray[np.int64]:
        return np.array([self.index[s.path] for s in samples], np.int64)


def _decode(path: Path, sizes: tuple[int, ...]) -> list[NDArray[np.uint8]]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"could not read {path}")
    return [np.asarray(cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA), np.uint8)
            for s in sizes]


def build_cache(root: Path, sizes: tuple[int, ...] = CACHE_SIZES, workers: int = 8,
                progress: Callable[[str], None] = print) -> None:
    raw = root / "raw" / "cnr-ext"
    samples = read_labels(raw / "LABELS" / "all.txt")
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    arrays = {
        s: np.lib.format.open_memmap(cache / f"patches_{s}.npy", "w+", np.uint8,
                                     (len(samples), s, s, 3))
        for s in sizes
    }
    paths = [raw / "PATCHES" / x.path for x in samples]
    with ThreadPoolExecutor(workers) as pool:
        for i, out in enumerate(pool.map(lambda p: _decode(p, sizes), paths, chunksize=256)):
            for s, arr in zip(sizes, out, strict=True):
                arrays[s][i] = arr
            if i % 20000 == 0:
                progress(f"decoded {i:,}/{len(samples):,}")
    for a in arrays.values():
        a.flush()
    with open(cache / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label"])
        w.writerows((x.path, x.label) for x in samples)
    progress(f"cached {len(samples):,} patches at sizes {sizes}")


def load_cache(root: Path, size: int) -> PatchSet:
    cache = root / "cache"
    with open(cache / "index.csv", newline="") as fh:
        samples = [parse_path(r["path"], int(r["label"])) for r in csv.DictReader(fh)]
    images = np.load(cache / f"patches_{size}.npy", mmap_mode="r")
    return PatchSet(np.asarray(images), samples, {s.path: i for i, s in enumerate(samples)})


# --------------------------------------------------------------------------- splits


@dataclass(frozen=True)
class Split:
    name: str
    description: str
    train: NDArray[np.int64]
    val: NDArray[np.int64]
    test: NDArray[np.int64]


UNSEEN_TRAIN_CAMERAS = frozenset({1, 2, 3, 4, 5, 6})
UNSEEN_TEST_CAMERAS = frozenset({7, 8, 9})


def benchmark_splits(ps: PatchSet, root: Path) -> list[Split]:
    """The three evaluation protocols in docs/benchmark.md.

    All three keep the dataset's day-level train/val/test partition, so no test image
    was taken on a day that appears in training.
    """
    labels = root / "raw" / "cnr-ext" / "LABELS"
    tr = read_labels(labels / "train.txt")
    va = read_labels(labels / "val.txt")
    te = read_labels(labels / "test.txt")

    def pick(xs: list[Sample], keep: Callable[[Sample], bool]) -> NDArray[np.int64]:
        return ps.rows(x for x in xs if keep(x))

    def every(_: Sample) -> bool:
        return True

    def seen(s: Sample) -> bool:
        return s.camera in UNSEEN_TRAIN_CAMERAS

    def unseen(s: Sample) -> bool:
        return s.camera in UNSEEN_TEST_CAMERAS

    def sunny(s: Sample) -> bool:
        return s.weather == "SUNNY"

    def cloudy_or_rain(s: Sample) -> bool:
        return s.weather != "SUNNY"

    return [
        Split("standard", "train, val and test days as published with the dataset",
              pick(tr, every), pick(va, every), pick(te, every)),
        Split("unseen-cameras", "train on cameras 1-6, test on cameras 7-9 (test days)",
              pick(tr, seen), pick(va, seen), pick(te, unseen)),
        Split("sun-to-cloud-and-rain", "train on sunny days, test on overcast and rainy days",
              pick(tr, sunny), pick(va, sunny), pick(te, cloudy_or_rain)),
    ]
