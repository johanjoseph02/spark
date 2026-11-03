"""Where frames come from: a camera, a video, an RTSP stream or a folder of stills."""

from __future__ import annotations

import calendar
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]

# 2015-11-22_0806.jpg (CNRPark+EXT) or 20151122-080600.jpg
_STAMP = re.compile(r"(\d{4})-?(\d{2})-?(\d{2})[_T-](\d{2})(\d{2})(\d{2})?")


def timestamp_from_name(name: str) -> float | None:
    m = _STAMP.search(name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(g) if g else 0 for g in m.groups())
    return float(calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)))


@dataclass
class FrameFolderSource:
    """Timestamped stills, replayed in time order at ``speed`` times real time.

    ``sparse`` is true: frames are minutes apart, so the agent should trust each one
    instead of waiting for a dwell time to pass.
    """

    folder: Path
    speed: float = 1.0
    loop: bool = False
    sparse: bool = True
    pattern: str = "**/*"
    max_gap_s: float = 3.0  # cap on real waiting time between frames (overnight gaps)

    def frames(self) -> list[tuple[float, Path]]:
        found = []
        for p in sorted(self.folder.glob(self.pattern)):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                t = timestamp_from_name(p.name)
                if t is not None:
                    found.append((t, p))
        found.sort()
        if not found:
            raise FileNotFoundError(f"no timestamped images under {self.folder}")
        return found

    def __iter__(self) -> Iterator[tuple[float, Frame]]:
        frames = self.frames()
        while True:
            prev_t: float | None = None
            for t, path in frames:
                if prev_t is not None and self.speed > 0:
                    time.sleep(min(max(t - prev_t, 0.0) / self.speed, self.max_gap_s))
                prev_t = t
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is not None:
                    yield t, np.asarray(img, np.uint8)
            if not self.loop:
                return


@dataclass
class OpenCVSource:
    """A camera index (``0``), a video file or a stream URL, stamped with wall-clock time."""

    spec: str
    fps: float = 2.0
    sparse: bool = False

    def __iter__(self) -> Iterator[tuple[float, Frame]]:
        target = int(self.spec) if self.spec.isdigit() else self.spec
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            raise OSError(f"could not open video source {self.spec!r}")
        is_file = isinstance(target, str) and Path(target).exists()
        native = cap.get(cv2.CAP_PROP_FPS) or 25.0
        every = max(1, round(native / self.fps)) if is_file else 1
        interval = 1.0 / self.fps
        n, last = 0, 0.0
        start = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                n += 1
                if is_file:
                    if (n - 1) % every:
                        continue
                    yield start + (n - 1) / native, np.asarray(frame, np.uint8)
                else:
                    now = time.time()
                    if now - last < interval:
                        continue
                    last = now
                    yield now, np.asarray(frame, np.uint8)
        finally:
            cap.release()


def open_source(spec: str, fps: float = 2.0, speed: float = 1.0,
                loop: bool = False) -> FrameFolderSource | OpenCVSource:
    p = Path(spec)
    if p.is_dir():
        return FrameFolderSource(p, speed=speed, loop=loop)
    return OpenCVSource(spec, fps=fps)
