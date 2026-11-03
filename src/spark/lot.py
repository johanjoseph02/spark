"""Lots, bays and the geometry that ties a bay to pixels in a camera frame."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

Point = tuple[float, float]


@dataclass(frozen=True)
class Bay:
    """One parking space as seen by one camera.

    ``polygon`` is in pixel coordinates of the camera frame the lot was mapped on
    (``Lot.frame_size``), clockwise or anticlockwise, at least three points.
    """

    id: str
    polygon: tuple[Point, ...]
    zone: str = "main"

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"bay {self.id}: a polygon needs at least 3 points")

    def bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def scaled(self, sx: float, sy: float) -> Bay:
        return Bay(self.id, tuple((x * sx, y * sy) for x, y in self.polygon), self.zone)


@dataclass(frozen=True)
class Lot:
    """A car park watched by one camera."""

    id: str
    name: str
    frame_size: tuple[int, int]
    bays: tuple[Bay, ...]
    lat: float | None = None
    lon: float | None = None
    meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [b.id for b in self.bays]
        if len(set(ids)) != len(ids):
            raise ValueError(f"lot {self.id}: bay ids must be unique")

    @property
    def capacity(self) -> int:
        return len(self.bays)

    def bay(self, bay_id: str) -> Bay:
        for b in self.bays:
            if b.id == bay_id:
                return b
        raise KeyError(bay_id)

    def zones(self) -> list[str]:
        return sorted({b.zone for b in self.bays})

    def for_frame(self, width: int, height: int) -> Lot:
        """The same lot, with polygons rescaled to a frame of another size."""
        fw, fh = self.frame_size
        if (fw, fh) == (width, height):
            return self
        sx, sy = width / fw, height / fh
        return Lot(
            self.id,
            self.name,
            (width, height),
            tuple(b.scaled(sx, sy) for b in self.bays),
            self.lat,
            self.lon,
            dict(self.meta),
        )

    # ---------------------------------------------------------------- persistence

    def to_json(self) -> str:
        data = asdict(self)
        data["bays"] = [
            {"id": b.id, "zone": b.zone, "polygon": [list(p) for p in b.polygon]}
            for b in self.bays
        ]
        return json.dumps(data, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Lot:
        bays_raw = data["bays"]
        assert isinstance(bays_raw, list)
        bays = tuple(
            Bay(
                str(b["id"]),
                tuple((float(x), float(y)) for x, y in b["polygon"]),
                str(b.get("zone", "main")),
            )
            for b in bays_raw
        )
        size = data["frame_size"]
        assert isinstance(size, list | tuple)
        meta = data.get("meta") or {}
        assert isinstance(meta, dict)
        lat, lon = data.get("lat"), data.get("lon")
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            frame_size=(int(size[0]), int(size[1])),
            bays=bays,
            lat=float(lat) if isinstance(lat, int | float) else None,
            lon=float(lon) if isinstance(lon, int | float) else None,
            meta={str(k): str(v) for k, v in meta.items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> Lot:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n")


def lot_from_cnrpark_csv(
    path: str | Path,
    lot_id: str,
    name: str,
    source_size: tuple[int, int] = (2592, 1944),
) -> Lot:
    """Read a CNRPark+EXT ``cameraN.csv`` (SlotId,X,Y,W,H).

    The boxes are given for the 2592x1944 originals; ``Lot.for_frame`` rescales them to
    the 1000x750 frames that ship with the dataset.
    """
    bays: list[Bay] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            x, y, w, h = (float(row[k]) for k in ("X", "Y", "W", "H"))
            bays.append(Bay(row["SlotId"], ((x, y), (x + w, y), (x + w, y + h), (x, y + h))))
    return Lot(lot_id, name, source_size, tuple(bays))
