"""SQLite-backed state: bay occupancy, change events, snapshots and prices.

Timestamps are the *observation* times sent by the edge agent, in seconds since the Unix
epoch, lot-local. Everything time-based (snapshots, price reviews, forecasts) runs on
those timestamps, not on the server clock, so replaying recorded footage at 600x speed
produces the same analytics as watching it live.
"""

from __future__ import annotations

import functools
import json
import sqlite3
import statistics
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from spark.lot import Lot
from spark.server.forecast import WEEK_HOURS, HourOfWeekProfile, hour_of_week
from spark.server.pricing import PriceDecision, Pricer, PricingConfig

SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS lots (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, capacity INTEGER NOT NULL,
    lat REAL, lon REAL, currency TEXT NOT NULL DEFAULT 'EUR', meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS bays (
    lot_id TEXT NOT NULL, bay_id TEXT NOT NULL, zone TEXT NOT NULL,
    occupied INTEGER, p REAL, since REAL, updated REAL,
    PRIMARY KEY (lot_id, bay_id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, lot_id TEXT NOT NULL, bay_id TEXT NOT NULL,
    occupied INTEGER NOT NULL, t REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_lot_t ON events (lot_id, t);
CREATE TABLE IF NOT EXISTS snapshots (
    lot_id TEXT NOT NULL, t REAL NOT NULL, occupied INTEGER NOT NULL,
    known INTEGER NOT NULL, capacity INTEGER NOT NULL,
    PRIMARY KEY (lot_id, t)
);
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY, lot_id TEXT NOT NULL, t REAL NOT NULL, price REAL NOT NULL,
    occupancy REAL, reason TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS prices_lot_t ON prices (lot_id, t);
CREATE TABLE IF NOT EXISTS price_bands (
    lot_id TEXT NOT NULL, band TEXT NOT NULL, price REAL NOT NULL,
    PRIMARY KEY (lot_id, band)
);
"""


@dataclass(frozen=True)
class BayReport:
    occupied: bool
    p: float = 1.0


class UnknownLot(KeyError):
    pass


def locked[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    """One sqlite3 connection is shared by the server's worker threads; every access to
    it goes through the store's re-entrant lock."""

    @functools.wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]
        with self._lock:  # type: ignore[attr-defined]
            return method(*args, **kwargs)

    return wrapper


class Store:
    def __init__(self, path: str, pricing: PricingConfig | None = None,
                 snapshot_every_s: float = 60.0) -> None:
        self.pricing = pricing or PricingConfig()
        self.snapshot_every_s = snapshot_every_s
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._lock = threading.RLock()
        self._profiles: dict[str, HourOfWeekProfile] = {}
        self._lots: dict[str, Lot] = {}
        self._last_snapshot: dict[str, float] = {}
        self._pricers: dict[str, Pricer] = {}

    @locked
    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------------------------ lots

    @locked
    def register_lot(self, lot: Lot, currency: str = "EUR") -> None:
        with self._lock:
            self._db.execute("BEGIN")
            self._db.execute(
                "INSERT INTO lots (id, name, capacity, lat, lon, currency, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,"
                " capacity=excluded.capacity, lat=excluded.lat, lon=excluded.lon,"
                " currency=excluded.currency, meta=excluded.meta",
                (lot.id, lot.name, lot.capacity, lot.lat, lot.lon, currency,
                 json.dumps(lot.meta)),
            )
            known = {r["bay_id"] for r in self._db.execute(
                "SELECT bay_id FROM bays WHERE lot_id = ?", (lot.id,))}
            for b in lot.bays:
                if b.id not in known:
                    self._db.execute(
                        "INSERT INTO bays (lot_id, bay_id, zone) VALUES (?, ?, ?)",
                        (lot.id, b.id, b.zone))
            ids = [b.id for b in lot.bays]
            self._db.execute(
                f"DELETE FROM bays WHERE lot_id = ? AND bay_id NOT IN ({','.join('?' * len(ids))})",
                (lot.id, *ids))
            self._db.execute("COMMIT")
            self._lots[lot.id] = lot
            self._profiles[lot.id] = self._rebuild_profile(lot.id)
            bands = {r["band"]: float(r["price"]) for r in self._db.execute(
                "SELECT band, price FROM price_bands WHERE lot_id = ?", (lot.id,))}
            self._pricers[lot.id] = Pricer(self.pricing, self.price_now(lot.id), bands)
            row = self._db.execute("SELECT MAX(t) AS t FROM snapshots WHERE lot_id = ?",
                                   (lot.id,)).fetchone()
            if row["t"] is not None:
                self._last_snapshot[lot.id] = float(row["t"])

    def lots(self) -> list[Lot]:
        return list(self._lots.values())

    def _lot(self, lot_id: str) -> Lot:
        try:
            return self._lots[lot_id]
        except KeyError:
            raise UnknownLot(lot_id) from None

    def _rebuild_profile(self, lot_id: str) -> HourOfWeekProfile:
        prof = HourOfWeekProfile()
        for r in self._hourly(lot_id):
            prof.observe(r[0], r[1])
        return prof

    def _hourly(self, lot_id: str, since: float | None = None) -> list[tuple[float, float]]:
        """Mean occupancy per clock hour, oldest first."""
        rows = self._db.execute(
            "SELECT CAST(t / 3600 AS INTEGER) AS h, AVG(CAST(occupied AS REAL) / known) AS occ "
            "FROM snapshots WHERE lot_id = ? AND known > 0 AND t >= ? GROUP BY h ORDER BY h",
            (lot_id, since if since is not None else float("-inf")),
        ).fetchall()
        return [(int(r["h"]) * 3600.0, float(r["occ"])) for r in rows]

    # ------------------------------------------------------------------ reports

    @locked
    def report(self, lot_id: str, t: float, bays: dict[str, BayReport]) -> dict[str, Any]:
        """Apply one observation from an edge agent. Returns what changed."""
        lot = self._lot(lot_id)
        valid = {b.id for b in lot.bays}
        changed: list[dict[str, Any]] = []
        with self._lock:
            current = {r["bay_id"]: r for r in self._db.execute(
                "SELECT bay_id, occupied, updated FROM bays WHERE lot_id = ?", (lot_id,))}
            self._db.execute("BEGIN")
            for bay_id, rep in bays.items():
                if bay_id not in valid:
                    continue
                row = current[bay_id]
                if row["updated"] is not None and t < row["updated"]:
                    continue  # stale, out-of-order report
                occ = 1 if rep.occupied else 0
                if row["occupied"] is None or row["occupied"] != occ:
                    self._db.execute(
                        "UPDATE bays SET occupied = ?, p = ?, since = ?, updated = ? "
                        "WHERE lot_id = ? AND bay_id = ?", (occ, rep.p, t, t, lot_id, bay_id))
                    self._db.execute(
                        "INSERT INTO events (lot_id, bay_id, occupied, t) VALUES (?, ?, ?, ?)",
                        (lot_id, bay_id, occ, t))
                    changed.append({"bay_id": bay_id, "occupied": bool(occ)})
                else:
                    self._db.execute(
                        "UPDATE bays SET p = ?, updated = ? WHERE lot_id = ? AND bay_id = ?",
                        (rep.p, t, lot_id, bay_id))
            snap = self._maybe_snapshot(lot, t)
            self._db.execute("COMMIT")
        decision = self.maybe_reprice(lot_id, t)
        return {
            "changed": changed,
            "snapshot": snap,
            "price": None if decision is None else decision.__dict__,
        }

    def _maybe_snapshot(self, lot: Lot, t: float) -> bool:
        last = self._last_snapshot.get(lot.id)
        if last is not None and t - last < self.snapshot_every_s:
            return False
        r = self._db.execute(
            "SELECT SUM(occupied = 1) AS occ, COUNT(occupied) AS known FROM bays "
            "WHERE lot_id = ?", (lot.id,)).fetchone()
        occ, known = int(r["occ"] or 0), int(r["known"] or 0)
        self._db.execute(
            "INSERT OR REPLACE INTO snapshots (lot_id, t, occupied, known, capacity) "
            "VALUES (?, ?, ?, ?, ?)", (lot.id, t, occ, known, lot.capacity))
        self._last_snapshot[lot.id] = t
        if known:
            self._profiles[lot.id].observe(t, occ / known)
        return True

    # ------------------------------------------------------------------ pricing

    @locked
    def price_now(self, lot_id: str) -> float:
        r = self._db.execute("SELECT price FROM prices WHERE lot_id = ? ORDER BY t DESC, id DESC "
                             "LIMIT 1", (lot_id,)).fetchone()
        return float(r["price"]) if r else self.pricing.base_price

    @locked
    def maybe_reprice(self, lot_id: str, now: float, force: bool = False) -> PriceDecision | None:
        cfg = self.pricing
        with self._lock:
            last = self._db.execute(
                "SELECT t FROM prices WHERE lot_id = ? ORDER BY t DESC, id DESC LIMIT 1",
                (lot_id,)).fetchone()
            if last is None:
                first = self._db.execute("SELECT MIN(t) AS t FROM snapshots WHERE lot_id = ?",
                                         (lot_id,)).fetchone()["t"]
                if first is None:
                    return None
                self._insert_price(lot_id, float(first), self._pricers[lot_id].price, None,
                                   "initial price")
                last_t = float(first)
            else:
                last_t = float(last["t"])
            if not force and now - last_t < cfg.review_every_s:
                return None
            measured = self._mean_occupancy(lot_id, now - cfg.review_every_s, now)
            if measured is None:
                return None
            current = self._mean_occupancy(lot_id, now - self.snapshot_every_s * 1.5, now)
            pricer = self._pricers[lot_id]
            decision = pricer.review(now, measured, measured if current is None else current,
                                     self._profiles[lot_id])
            self._db.execute("BEGIN")
            self._insert_price(lot_id, now, decision.price, decision.occupancy, decision.reason)
            self._db.executemany(
                "INSERT OR REPLACE INTO price_bands (lot_id, band, price) VALUES (?, ?, ?)",
                [(lot_id, b, p) for b, p in pricer.bands.items()])
            self._db.execute("COMMIT")
            return decision

    def _insert_price(self, lot_id: str, t: float, price: float, occ: float | None,
                      reason: str) -> None:
        self._db.execute(
            "INSERT INTO prices (lot_id, t, price, occupancy, reason) VALUES (?, ?, ?, ?, ?)",
            (lot_id, t, price, occ, reason))

    def _mean_occupancy(self, lot_id: str, since: float, until: float) -> float | None:
        r = self._db.execute(
            "SELECT AVG(CAST(occupied AS REAL) / known) AS occ FROM snapshots "
            "WHERE lot_id = ? AND known > 0 AND t > ? AND t <= ?", (lot_id, since, until)
        ).fetchone()
        return None if r["occ"] is None else float(r["occ"])

    @locked
    def price_bands(self, lot_id: str) -> dict[str, float]:
        self._lot(lot_id)
        return dict(sorted(self._pricers[lot_id].bands.items()))

    @locked
    def price_history(self, lot_id: str, limit: int = 500) -> list[dict[str, Any]]:
        self._lot(lot_id)
        rows = self._db.execute(
            "SELECT t, price, occupancy, reason FROM prices WHERE lot_id = ? "
            "ORDER BY t DESC, id DESC LIMIT ?", (lot_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------ reads

    @locked
    def status(self, lot_id: str) -> dict[str, Any]:
        lot = self._lot(lot_id)
        meta = self._db.execute("SELECT * FROM lots WHERE id = ?", (lot_id,)).fetchone()
        rows = self._db.execute(
            "SELECT bay_id, zone, occupied, p, since, updated FROM bays WHERE lot_id = ?",
            (lot_id,)).fetchall()
        zones: dict[str, dict[str, int]] = defaultdict(lambda: {"capacity": 0, "free": 0,
                                                               "occupied": 0, "unknown": 0})
        for r in rows:
            z = zones[r["zone"]]
            z["capacity"] += 1
            key = "unknown" if r["occupied"] is None else ("occupied" if r["occupied"] else "free")
            z[key] += 1
        updated = max((r["updated"] for r in rows if r["updated"] is not None), default=None)
        free = sum(z["free"] for z in zones.values())
        occupied = sum(z["occupied"] for z in zones.values())
        return {
            "id": lot.id,
            "name": lot.name,
            "lat": lot.lat,
            "lon": lot.lon,
            "capacity": lot.capacity,
            "free": free,
            "occupied": occupied,
            "unknown": lot.capacity - free - occupied,
            "zones": dict(zones),
            "updated": updated,
            "price": self.price_now(lot_id),
            "currency": meta["currency"],
            "bays": [
                {"id": r["bay_id"], "zone": r["zone"],
                 "occupied": None if r["occupied"] is None else bool(r["occupied"]),
                 "p": r["p"], "since": r["since"]}
                for r in rows
            ],
        }

    @locked
    def history(self, lot_id: str, since: float | None = None,
                limit: int = 5000) -> list[dict[str, Any]]:
        self._lot(lot_id)
        rows = self._db.execute(
            "SELECT t, occupied, known, capacity FROM snapshots WHERE lot_id = ? AND t >= ? "
            "ORDER BY t DESC LIMIT ?",
            (lot_id, since if since is not None else float("-inf"), limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    @locked
    def analytics(self, lot_id: str) -> dict[str, Any]:
        """What an operator or planner needs: when the lot fills, how long cars stay,
        how hard each bay works."""
        lot = self._lot(lot_id)
        profile: list[list[float | None]] = [[None] * 24 for _ in range(7)]
        sums: dict[int, list[float]] = defaultdict(list)
        for t, occ in self._hourly(lot_id):
            sums[hour_of_week(t)].append(occ)
        for h in range(WEEK_HOURS):
            if sums[h]:
                profile[h // 24][h % 24] = round(statistics.fmean(sums[h]), 4)

        events = self._db.execute(
            "SELECT bay_id, occupied, t FROM events WHERE lot_id = ? ORDER BY bay_id, t",
            (lot_id,)).fetchall()
        dwell: list[float] = []
        arrivals: dict[str, int] = defaultdict(int)
        start: dict[str, float] = {}
        for e in events:
            if e["occupied"]:
                start[e["bay_id"]] = e["t"]
                arrivals[e["bay_id"]] += 1
            elif e["bay_id"] in start:
                dwell.append((e["t"] - start.pop(e["bay_id"])) / 60.0)
        span = self._db.execute("SELECT MIN(t) AS a, MAX(t) AS b FROM snapshots WHERE lot_id = ?",
                                (lot_id,)).fetchone()
        days = max(((span["b"] or 0) - (span["a"] or 0)) / 86400.0, 1 / 24)
        peak = self._db.execute(
            "SELECT t, CAST(occupied AS REAL) / known AS occ FROM snapshots "
            "WHERE lot_id = ? AND known > 0 ORDER BY occ DESC, t LIMIT 1", (lot_id,)).fetchone()
        full_share = self._db.execute(
            "SELECT AVG(CAST(occupied AS REAL) / known >= 0.95) AS s FROM snapshots "
            "WHERE lot_id = ? AND known > 0", (lot_id,)).fetchone()["s"]
        return {
            "hour_of_week": profile,
            "dwell_minutes": _summary(dwell),
            "arrivals": sum(arrivals.values()),
            "turnover_per_bay_per_day": round(sum(arrivals.values()) / lot.capacity / days, 3),
            "busiest_bays": sorted(arrivals.items(), key=lambda kv: -kv[1])[:5],
            "peak": None if peak is None else {"t": peak["t"], "occupancy": peak["occ"]},
            "share_of_time_full": None if full_share is None else round(float(full_share), 4),
            "days_observed": round(days, 2),
        }


def _summary(xs: list[float]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "median": None, "p90": None, "mean": None}
    s = sorted(xs)
    return {
        "n": len(s),
        "median": round(statistics.median(s), 1),
        "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 1),
        "mean": round(statistics.fmean(s), 1),
    }
