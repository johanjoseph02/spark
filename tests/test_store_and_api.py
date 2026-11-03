from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from spark.lot import Lot
from spark.server.app import Hub, create_app
from spark.server.pricing import PricingConfig
from spark.server.settings import Settings
from spark.server.store import BayReport, Store, UnknownLot

T0 = 1_704_096_000.0  # Mon 1 Jan 2024 08:00 UTC


def all_bays(lot: Lot, occupied: set[str]) -> dict[str, BayReport]:
    return {b.id: BayReport(b.id in occupied, 0.9) for b in lot.bays}


@pytest.fixture
def store(tmp_path: Path, lot: Lot) -> Store:
    s = Store(str(tmp_path / "s.db"), PricingConfig(review_every_s=3600), snapshot_every_s=60)
    s.register_lot(lot)
    return s


def test_report_records_changes_and_status(store: Store, lot: Lot) -> None:
    out = store.report(lot.id, T0, all_bays(lot, {"B0"}))
    assert len(out["changed"]) == 4 and out["snapshot"]
    st = store.status(lot.id)
    assert (st["free"], st["occupied"], st["unknown"]) == (3, 1, 0)
    assert st["zones"]["a"] == {"capacity": 2, "free": 1, "occupied": 1, "unknown": 0}

    out = store.report(lot.id, T0 + 10, all_bays(lot, {"B0", "B1"}))
    assert out["changed"] == [{"bay_id": "B1", "occupied": True}]
    assert not out["snapshot"]  # within the snapshot interval


def test_stale_and_unknown_bays_are_ignored(store: Store, lot: Lot) -> None:
    store.report(lot.id, T0 + 100, all_bays(lot, set()))
    store.report(lot.id, T0, {"B0": BayReport(True), "nope": BayReport(True)})
    assert store.status(lot.id)["free"] == 4
    with pytest.raises(UnknownLot):
        store.report("missing", T0, {})


def test_price_reviews_run_on_observation_time(tmp_path: Path, lot: Lot) -> None:
    store = Store(str(tmp_path / "r.db"), PricingConfig(mode="responsive"))
    store.register_lot(lot)
    busy = {b.id for b in lot.bays}
    for k in range(0, 61):
        store.report(lot.id, T0 + k * 60, all_bays(lot, busy))
    hist = store.price_history(lot.id)
    assert hist[0]["reason"] == "initial price"
    assert hist[-1]["price"] > 2.0  # full for an hour: price went up
    assert "above" in hist[-1]["reason"]
    assert store.price_now(lot.id) == hist[-1]["price"]


def test_banded_prices_learn_per_hour_and_persist(tmp_path: Path, lot: Lot) -> None:
    store = Store(str(tmp_path / "b.db"), PricingConfig(mode="banded"))
    store.register_lot(lot)
    busy = {b.id for b in lot.bays}
    for k in range(0, 121):  # Monday 08:00 to 10:00, full the whole time
        store.report(lot.id, T0 + k * 60, all_bays(lot, busy))
    bands = store.price_bands(lot.id)
    assert bands["weekday-08"] > 2.0 and bands["weekday-09"] > 2.0
    assert store.price_now(lot.id) == 2.0  # 10:00 has no history yet: base price
    store.close()
    again = Store(str(tmp_path / "b.db"), PricingConfig(mode="banded"))
    again.register_lot(lot)
    assert again.price_bands(lot.id) == bands


def test_analytics_dwell_and_turnover(store: Store, lot: Lot) -> None:
    store.report(lot.id, T0, all_bays(lot, set()))
    store.report(lot.id, T0 + 600, all_bays(lot, {"B0"}))
    store.report(lot.id, T0 + 600 + 45 * 60, all_bays(lot, set()))
    a = store.analytics(lot.id)
    assert a["dwell_minutes"]["n"] == 1
    assert a["dwell_minutes"]["median"] == 45.0
    assert a["arrivals"] == 1
    assert a["hour_of_week"][0][8] is not None  # Monday 08:00


def test_concurrent_reports_and_reads(store: Store, lot: Lot) -> None:
    import threading

    errors: list[BaseException] = []

    def writer(offset: int) -> None:
        try:
            for k in range(200):
                busy = {"B0"} if k % 2 else set()
                store.report(lot.id, T0 + offset + k * 30, all_bays(lot, busy))
        except BaseException as e:
            errors.append(e)

    def reader() -> None:
        try:
            for _ in range(200):
                store.status(lot.id)
                store.history(lot.id)
                store.analytics(lot.id)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_state_survives_restart(tmp_path: Path, lot: Lot) -> None:
    s = Store(str(tmp_path / "s.db"))
    s.register_lot(lot)
    s.report(lot.id, T0, all_bays(lot, {"B2"}))
    s.close()
    s2 = Store(str(tmp_path / "s.db"))
    s2.register_lot(lot)
    assert s2.status(lot.id)["occupied"] == 1
    assert s2.history(lot.id)[0]["occupied"] == 1


# ---------------------------------------------------------------------- HTTP


@pytest.fixture
def client(tmp_path: Path, lot: Lot) -> TestClient:
    lots = tmp_path / "lots"
    lots.mkdir()
    lot.save(lots / "test-lot.json")
    gtfs = tmp_path / "gtfs"
    gtfs.mkdir()
    (gtfs / "stops.txt").write_text("stop_id,stop_name,stop_lat,stop_lon\nS,Stop,43.7201,10.4201\n")
    settings = Settings(db_path=str(tmp_path / "api.db"), lots_dir=str(lots),
                        device_token="dev-secret", admin_token="op-secret", gtfs_dir=str(gtfs))
    return TestClient(create_app(settings))


def report(client: TestClient, t: float, occupied: set[str], token: str = "dev-secret"):
    body = {"t": t, "bays": {f"B{i}": {"occupied": f"B{i}" in occupied, "p": 0.8}
                             for i in range(4)}}
    return client.post("/api/lots/test-lot/report", json=body,
                       headers={"Authorization": f"Bearer {token}"})


def test_reports_need_the_device_token(client: TestClient) -> None:
    assert report(client, T0, set(), token="wrong").status_code == 401
    r = client.post("/api/lots/test-lot/report", json={"t": T0, "bays": {}})
    assert r.status_code == 401
    assert report(client, T0, {"B1"}).status_code == 200


def test_read_endpoints(client: TestClient) -> None:
    report(client, T0, {"B1"})
    lots = client.get("/api/lots").json()
    assert lots[0]["free"] == 3 and lots[0]["currency"] == "EUR"
    assert client.get("/api/lots/test-lot").json()["bays"][1]["occupied"] is True
    assert len(client.get("/api/lots/test-lot/geometry").json()["bays"]) == 4
    assert client.get("/api/lots/test-lot/history").json()[0]["occupied"] == 1
    assert client.get("/api/lots/test-lot/analytics").json()["days_observed"] > 0
    assert client.get("/api/lots/test-lot/transit").json()[0]["name"] == "Stop"
    near = client.get("/api/nearby", params={"lat": 43.72, "lon": 10.42}).json()
    assert near[0]["id"] == "test-lot" and near[0]["distance_m"] == 0
    assert client.get("/api/nearby", params={"lat": 0, "lon": 0}).json() == []
    assert client.get("/api/lots/nope").status_code == 404
    assert client.get("/healthz").json() == {"ok": True, "lots": 1}


def test_operator_review_needs_admin_token(client: TestClient) -> None:
    url = "/api/lots/test-lot/prices/review"
    assert client.post(url, headers={"Authorization": "Bearer op-secret"}).status_code == 409
    report(client, T0, {"B0", "B1", "B2", "B3"})
    assert client.post(url).status_code == 401
    r = client.post(url, headers={"Authorization": "Bearer op-secret"})
    assert r.status_code == 200 and r.json()["decision"] is not None
    prices = client.get("/api/lots/test-lot/prices").json()
    assert prices["current"] == r.json()["decision"]["price"]
    assert prices["bands"]["weekday-07"] > 2.0  # the hour that just ended was full


def test_web_apps_are_served(client: TestClient) -> None:
    assert "SPARK" in client.get("/").text
    assert "operator" in client.get("/admin").text
    assert "map a car park" in client.get("/mapper").text
    assert client.get("/static/driver.js").status_code == 200
    assert client.get("/sw.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/manifest.webmanifest").json()["short_name"] == "SPARK"


def test_hub_fans_out_to_subscribers() -> None:
    async def scenario() -> list[str]:
        hub = Hub()
        hub.loop = asyncio.get_running_loop()
        a, b = hub.subscribe(), hub.subscribe()
        hub.publish("lot", {"id": "x", "free": 3})
        await asyncio.sleep(0)
        hub.unsubscribe(b)
        hub.publish("lot", {"id": "x", "free": 2})
        await asyncio.sleep(0)
        return [a.get_nowait(), a.get_nowait(), b.get_nowait()] + (
            ["leak"] if not b.empty() else [])

    msgs = asyncio.run(scenario())
    assert msgs[0].startswith("event: lot\ndata: ")
    assert '"free":2' in msgs[1]
    assert len(msgs) == 3
