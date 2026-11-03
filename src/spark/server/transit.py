"""Public-transport stops near a lot, from a standard GTFS feed.

Only ``stops.txt`` is required. If ``routes.txt``, ``trips.txt`` and ``stop_times.txt``
are present, each stop also lists the routes that serve it, so a driver turned away from
a full lot can see where to leave the car and continue by bus or train.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

WALK_M_PER_MIN = 80.0


@dataclass(frozen=True)
class Stop:
    id: str
    name: str
    lat: float
    lon: float
    routes: tuple[str, ...] = field(default=())


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_gtfs(folder: str | Path) -> list[Stop]:
    folder = Path(folder)
    stops = _rows(folder / "stops.txt")
    serving: dict[str, set[str]] = defaultdict(set)
    if all((folder / f).exists() for f in ("routes.txt", "trips.txt", "stop_times.txt")):
        names = {r["route_id"]: (r.get("route_short_name") or r.get("route_long_name") or
                                 r["route_id"]) for r in _rows(folder / "routes.txt")}
        trip_route = {r["trip_id"]: names.get(r["route_id"], r["route_id"])
                      for r in _rows(folder / "trips.txt")}
        for r in _rows(folder / "stop_times.txt"):
            route = trip_route.get(r["trip_id"])
            if route:
                serving[r["stop_id"]].add(route)
    return [
        Stop(s["stop_id"], s["stop_name"], float(s["stop_lat"]), float(s["stop_lon"]),
             tuple(sorted(serving.get(s["stop_id"], ()))))
        for s in stops
        if s.get("location_type", "0") in ("", "0") and s.get("stop_lat")
    ]


def nearby(stops: list[Stop], lat: float, lon: float, radius_m: float = 800.0,
           limit: int = 5) -> list[dict[str, object]]:
    found = []
    for s in stops:
        d = haversine_m(lat, lon, s.lat, s.lon)
        if d <= radius_m:
            found.append({
                "id": s.id, "name": s.name, "lat": s.lat, "lon": s.lon,
                "routes": list(s.routes), "distance_m": round(d),
                "walk_min": max(1, round(d / WALK_M_PER_MIN)),
            })
    found.sort(key=lambda x: float(str(x["distance_m"])))
    return found[:limit]
