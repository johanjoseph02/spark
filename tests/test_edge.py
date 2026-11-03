from __future__ import annotations

from pathlib import Path

import cv2
import httpx
import numpy as np

from spark.cli import build_parser
from spark.edge.agent import Agent, Publisher
from spark.edge.sources import FrameFolderSource, open_source, timestamp_from_name
from spark.lot import Lot
from spark.vision.temporal import FilterConfig


class Fake:
    """Occupied if the patch is bright."""

    name = "fake"
    input_size = 16

    def predict_proba(self, patches: np.ndarray) -> np.ndarray:
        return (patches.mean(axis=(1, 2, 3)) / 255.0).astype(np.float32)


def test_timestamps_from_filenames() -> None:
    assert timestamp_from_name("2015-11-22_0806.jpg") == 1448179560.0
    assert timestamp_from_name("cam_20240101-093015.png") == 1704101415.0
    assert timestamp_from_name("frame.jpg") is None


def test_folder_source_orders_by_time(tmp_path: Path) -> None:
    for name in ("2015-11-22_0900.jpg", "2015-11-22_0806.jpg", "notes.txt"):
        if name.endswith(".jpg"):
            cv2.imwrite(str(tmp_path / name), np.zeros((4, 4, 3), np.uint8))
        else:
            (tmp_path / name).write_text("x")
    src = open_source(str(tmp_path), speed=0)
    assert isinstance(src, FrameFolderSource)
    ts = [t for t, _ in src]
    assert ts == sorted(ts) and len(ts) == 2


def test_publisher_queues_while_offline_and_flushes_in_order() -> None:
    seen: list[float] = []
    online = {"up": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if not online["up"]:
            raise httpx.ConnectError("down")
        seen.append(__import__("json").loads(request.content)["t"])
        return httpx.Response(200, json={})

    pub = Publisher("http://x", "lot", client=httpx.Client(transport=httpx.MockTransport(handler)))
    pub.publish({"t": 1.0})
    pub.publish({"t": 2.0})
    assert len(pub.queue) == 2 and seen == []
    online["up"] = True
    pub._backoff_until = 0
    pub.publish({"t": 3.0})
    assert seen == [1.0, 2.0, 3.0] and not pub.queue and pub.sent == 3


def test_publisher_drops_rejected_reports() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(422, text="bad"))
    pub = Publisher("http://x", "lot", client=httpx.Client(transport=transport))
    pub.publish({"t": 1.0})
    assert not pub.queue and pub.sent == 0


def test_agent_reports_filtered_states(lot: Lot) -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(__import__("json").loads(request.content))
        return httpx.Response(200, json={})

    pub = Publisher("http://x", lot.id, client=httpx.Client(transport=httpx.MockTransport(handler)))
    agent = Agent(lot, Fake(), pub, FilterConfig(tau_s=0.5, min_dwell_s=0.0))
    frame = np.zeros((100, 200, 3), np.uint8)
    x0, y0 = (int(v) for v in lot.bays[2].polygon[0])
    frame[y0:y0 + 20, x0:x0 + 20] = 255  # a bright "car" in bay B2
    agent.step(0.0, frame)
    assert bodies[-1]["bays"]["B2"]["occupied"] is True
    assert bodies[-1]["bays"]["B0"]["occupied"] is False
    assert bodies[-1]["model"] == "fake"


def test_cli_parses_every_command() -> None:
    p = build_parser()
    for argv in (["data", "cache"], ["bench", "report"], ["serve"], ["demo", "--synthetic"],
                 ["sim", "--days", "1"], ["edge", "--lot", "x.json", "--source", "0"]):
        assert p.parse_args(argv).func
