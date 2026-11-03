"""HTTP API, live stream and the two web apps (driver and operator)."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from spark.lot import Lot
from spark.server.settings import Settings
from spark.server.store import BayReport, Store, UnknownLot
from spark.server.transit import Stop, haversine_m, load_gtfs, nearby

log = logging.getLogger("spark.server")
STATIC = Path(__file__).parent / "static"


class BayIn(BaseModel):
    occupied: bool
    p: float = Field(1.0, ge=0.0, le=1.0)


class ReportIn(BaseModel):
    t: float = Field(..., description="observation time, seconds since epoch, lot-local")
    bays: dict[str, BayIn]
    model: str = ""
    source: str = ""


class Hub:
    """Fan-out of lot updates to every open event stream."""

    def __init__(self) -> None:
        self._subs: set[asyncio.Queue[str]] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._subs.discard(q)

    def publish(self, event: str, data: dict[str, Any]) -> None:
        msg = f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
        loop = self.loop
        if loop is None:
            return
        for q in list(self._subs):
            loop.call_soon_threadsafe(_offer, q, msg)


def _offer(q: asyncio.Queue[str], msg: str) -> None:
    if q.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            q.get_nowait()  # drop the oldest message for a slow client
    q.put_nowait(msg)


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        return ""
    return authorization[7:].strip()


def create_app(settings: Settings | None = None, store: Store | None = None) -> FastAPI:
    settings = settings or Settings()
    store = store or Store(settings.db_path, settings.pricing, settings.snapshot_every_s)
    hub = Hub()
    stops: list[Stop] = load_gtfs(settings.gtfs_dir) if settings.gtfs_dir else []

    lots_dir = Path(settings.lots_dir)
    if lots_dir.is_dir():
        for path in sorted(lots_dir.glob("*.json")):
            lot = Lot.load(path)
            store.register_lot(lot, lot.meta.get("currency", "EUR"))
            log.info("registered lot %s (%d bays)", lot.id, lot.capacity)
    if not settings.device_token:
        log.warning("SPARK_DEVICE_TOKEN is not set: any client may post occupancy reports")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        hub.loop = asyncio.get_running_loop()
        yield
        hub.loop = None

    app = FastAPI(title="SPARK", version="0.2.0", lifespan=lifespan)
    app.state.store = store
    app.state.hub = hub

    def require(token: str) -> Any:
        def check(authorization: Annotated[str | None, Header()] = None) -> None:
            if token and not hmac.compare_digest(_bearer(authorization), token):
                raise HTTPException(401, "missing or wrong bearer token")
        return check

    device_auth = Depends(require(settings.device_token))
    admin_auth = Depends(require(settings.admin_token))

    def lot_or_404(lot_id: str) -> dict[str, Any]:
        try:
            return store.status(lot_id)
        except UnknownLot:
            raise HTTPException(404, f"no lot {lot_id!r}") from None

    def summary(s: dict[str, Any]) -> dict[str, Any]:
        return {k: s[k] for k in ("id", "name", "lat", "lon", "capacity", "free", "occupied",
                                  "unknown", "updated", "price", "currency")}

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "lots": len(store.lots())}

    @app.get("/api/lots")
    def list_lots() -> list[dict[str, Any]]:
        return [summary(store.status(lot.id)) for lot in store.lots()]

    @app.get("/api/lots/{lot_id}")
    def get_lot(lot_id: str) -> dict[str, Any]:
        return lot_or_404(lot_id)

    @app.get("/api/lots/{lot_id}/geometry")
    def geometry(lot_id: str) -> dict[str, Any]:
        lot_or_404(lot_id)
        lot = next(x for x in store.lots() if x.id == lot_id)
        data: dict[str, Any] = json.loads(lot.to_json())
        return data

    @app.post("/api/lots/{lot_id}/report", dependencies=[device_auth])
    def report(lot_id: str, body: ReportIn) -> dict[str, Any]:
        try:
            out = store.report(lot_id, body.t, {k: BayReport(v.occupied, v.p)
                                                for k, v in body.bays.items()})
        except UnknownLot:
            raise HTTPException(404, f"no lot {lot_id!r}") from None
        if out["changed"] or out["price"] or out["snapshot"]:
            hub.publish("lot", {**summary(store.status(lot_id)), "changed": out["changed"]})
        return out

    @app.get("/api/lots/{lot_id}/history")
    def history(lot_id: str, since: float | None = None,
                limit: Annotated[int, Query(le=20000)] = 5000) -> list[dict[str, Any]]:
        lot_or_404(lot_id)
        return store.history(lot_id, since, limit)

    @app.get("/api/lots/{lot_id}/analytics")
    def analytics(lot_id: str) -> dict[str, Any]:
        lot_or_404(lot_id)
        return store.analytics(lot_id)

    @app.get("/api/lots/{lot_id}/prices")
    def prices(lot_id: str) -> dict[str, Any]:
        s = lot_or_404(lot_id)
        return {"current": s["price"], "currency": s["currency"],
                "policy": settings.pricing.__dict__, "bands": store.price_bands(lot_id),
                "history": store.price_history(lot_id)}

    @app.post("/api/lots/{lot_id}/prices/review", dependencies=[admin_auth])
    def review(lot_id: str) -> dict[str, Any]:
        s = lot_or_404(lot_id)
        if s["updated"] is None:
            raise HTTPException(409, "no observations yet")
        d = store.maybe_reprice(lot_id, float(s["updated"]), force=True)
        if d is not None:
            hub.publish("lot", summary(store.status(lot_id)))
        return {"decision": None if d is None else d.__dict__}

    @app.get("/api/lots/{lot_id}/transit")
    def transit(lot_id: str, radius_m: float = 800.0) -> list[dict[str, object]]:
        s = lot_or_404(lot_id)
        if s["lat"] is None or s["lon"] is None:
            return []
        return nearby(stops, s["lat"], s["lon"], radius_m)

    @app.get("/api/nearby")
    def nearby_lots(lat: float, lon: float, radius_m: float = 3000.0) -> list[dict[str, Any]]:
        out = []
        for lot in store.lots():
            if lot.lat is None or lot.lon is None:
                continue
            d = haversine_m(lat, lon, lot.lat, lot.lon)
            if d <= radius_m:
                out.append({**summary(store.status(lot.id)), "distance_m": round(d)})
        return sorted(out, key=lambda x: (x["free"] == 0, x["distance_m"]))

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        q = hub.subscribe()

        async def gen() -> AsyncIterator[str]:
            try:
                yield "retry: 3000\n\n"
                while not await request.is_disconnected():
                    try:
                        yield await asyncio.wait_for(q.get(), timeout=15)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                hub.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/", include_in_schema=False)
        def driver() -> FileResponse:
            return FileResponse(STATIC / "driver.html")

        @app.get("/admin", include_in_schema=False)
        def admin() -> FileResponse:
            return FileResponse(STATIC / "admin.html")

        @app.get("/mapper", include_in_schema=False)
        def mapper() -> FileResponse:
            return FileResponse(STATIC / "mapper.html")

        @app.get("/sw.js", include_in_schema=False)
        def service_worker() -> FileResponse:
            return FileResponse(STATIC / "sw.js", media_type="text/javascript")

        @app.get("/manifest.webmanifest", include_in_schema=False)
        def manifest() -> FileResponse:
            return FileResponse(STATIC / "manifest.webmanifest",
                                media_type="application/manifest+json")

    return app
