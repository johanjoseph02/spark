"""Everything in one process: server, edge agent and the two web apps.

    uv run spark demo              # replay a CNRPark+EXT camera
                                   # (needs `spark data download --full`)
    uv run spark demo --synthetic  # simulated cars, no dataset needed

The real demo replays one camera's stills (taken roughly every 30 minutes over 23 days in
2015 and 2016) through the same agent a deployed camera would run, at 600x real time by
default. The synthetic demo skips vision and drives occupancy from the demand model in
``spark.sim``, which is useful for looking at the dashboard and pricing.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import uvicorn

from spark.edge.agent import Agent, Publisher
from spark.edge.sources import FrameFolderSource
from spark.lot import Bay, Lot, lot_from_cnrpark_csv
from spark.server.app import create_app
from spark.server.settings import Settings
from spark.sim.demand import DemandConfig, draw_arrivals
from spark.vision.classifiers import load_classifier
from spark.vision.temporal import FilterConfig

REPO = Path(__file__).resolve().parents[2]
CNR_LAT, CNR_LON = 43.7193, 10.4236  # CNR research area, Pisa (approximate)
log = logging.getLogger("spark.demo")


def _default_model() -> Path:
    for name in ("default.onnx", "hog-linear.npz", "edge-energy.npz"):
        p = REPO / "models" / name
        if p.exists():
            return p
    raise FileNotFoundError("no model in models/; run `spark bench run` or pull the repo again")


def _serve(settings: Settings, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=0.5).status_code == 200:
                return server
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError("server did not start")


def _synthetic_lot() -> Lot:
    bays = []
    for row in range(4):
        for col in range(12):
            x, y = 40 + col * 78, 30 + row * 150 + (20 if row % 2 else 0)
            bays.append(Bay(f"{'ABCD'[row]}{col + 1:02d}",
                            ((x, y), (x + 66, y), (x + 66, y + 110), (x, y + 110)),
                            zone="north" if row < 2 else "south"))
    return Lot("demo-sim", "Simulated lot", (1000, 640), tuple(bays), CNR_LAT + 0.004,
               CNR_LON - 0.003, {"currency": "EUR", "source": "simulated"})


def _run_synthetic(port: int, speed: float, days: float | None = None) -> None:
    lot = _synthetic_lot()
    client = httpx.Client(timeout=5.0)
    url = f"http://127.0.0.1:{port}/api/lots/{lot.id}/report"
    rng = np.random.default_rng(0)
    demand = DemandConfig(capacity=lot.capacity)
    parked: dict[str, float] = {}
    price = 2.0
    t = 1_700_000_000.0 - (1_700_000_000.0 % 604800) + 4 * 86400  # a Monday 00:00 UTC
    week0 = t
    step = 300.0
    while days is None or t < week0 + days * 86400:
        parked = {b: d for b, d in parked.items() if d > t}
        wtp, stay = draw_arrivals(demand, t - week0, step, rng)
        free = [b.id for b in lot.bays if b.id not in parked]
        rng.shuffle(free)
        for w, s in zip(wtp, stay, strict=True):
            if w >= price and free:
                parked[free.pop()] = t + s
        body = {"t": t, "model": "simulated",
                "bays": {b.id: {"occupied": b.id in parked, "p": 1.0} for b in lot.bays}}
        try:
            r = client.post(url, json=body)
            if r.is_success:
                lot_url = f"http://127.0.0.1:{port}/api/lots/{lot.id}"
                price = float(client.get(lot_url).json()["price"])
            else:
                log.warning("report rejected: %s %s", r.status_code, r.text[:200])
        except httpx.HTTPError as e:
            log.warning("server unreachable: %s", e)
        t += step
        time.sleep(step / speed)
    log.info("simulated %.0f days; the server stays up", days)
    while True:
        time.sleep(3600)


def run_demo(port: int = 8000, synthetic: bool = False, root: Path = Path("data"),
             camera: int = 8, speed: float = 600.0, days: float | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    tmp = Path(tempfile.mkdtemp(prefix="spark-demo-"))
    lots = tmp / "lots"
    lots.mkdir()
    gtfs = REPO / "configs" / "gtfs-example"
    if synthetic:
        _synthetic_lot().save(lots / "demo-sim.json")
    else:
        csv_path = root / "raw" / "full" / f"camera{camera}.csv"
        frames = root / "raw" / "full" / "FULL_IMAGE_1000x750"
        if not csv_path.exists() or not frames.exists():
            print("Full frames not found. Run `uv run spark data download --full` first, "
                  "or try `uv run spark demo --synthetic`.")
            return 1
        lot = lot_from_cnrpark_csv(csv_path, f"cnr-camera{camera}",
                                   f"CNR Pisa, camera {camera}").for_frame(1000, 750)
        lot = Lot(lot.id, lot.name, lot.frame_size, lot.bays, CNR_LAT, CNR_LON,
                  {"currency": "EUR", "source": "CNRPark+EXT replay"})
        lot.save(lots / f"{lot.id}.json")
    settings = Settings(db_path=str(tmp / "spark.db"), lots_dir=str(lots),
                        gtfs_dir=str(gtfs) if gtfs.exists() else "")
    _serve(settings, port)
    print(f"\n  Driver app:          http://127.0.0.1:{port}/"
          f"\n  Operator dashboard:  http://127.0.0.1:{port}/admin"
          f"\n  API docs:            http://127.0.0.1:{port}/docs\n"
          f"\n  Ctrl-C to stop.\n", flush=True)
    try:
        if synthetic:
            _run_synthetic(port, speed, days)
        else:
            clf = load_classifier(_default_model())
            source = FrameFolderSource(frames, speed=speed, loop=True,
                                       pattern=f"*/*/camera{camera}/*.jpg")
            agent = Agent(lot, clf, Publisher(f"http://127.0.0.1:{port}", lot.id),
                          FilterConfig(tau_s=1.0, min_dwell_s=0.0))
            log.info("replaying camera %d with %s", camera, clf.name)
            agent.run(source)
    except KeyboardInterrupt:
        pass
    return 0
