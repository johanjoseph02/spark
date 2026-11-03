"""Capture, classify, smooth, publish.

The agent keeps working when the network does not: reports queue in memory (bounded)
and are flushed oldest first once the server answers again. The server ignores reports
older than what it already has for a bay, so a late flush cannot roll the state back.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from spark.edge.sources import Frame, open_source
from spark.lot import Lot
from spark.vision.classifiers import OccupancyClassifier, load_classifier
from spark.vision.patches import extract_patches
from spark.vision.temporal import BayFilter, FilterConfig

log = logging.getLogger("spark.edge")


@dataclass
class AgentConfig:
    server: str
    lot_path: Path
    model_path: Path
    source: str
    token: str = ""
    fps: float = 2.0
    speed: float = 1.0
    loop: bool = False
    max_queue: int = 2000


class Publisher:
    def __init__(self, server: str, lot_id: str, token: str = "", max_queue: int = 2000,
                 client: httpx.Client | None = None) -> None:
        self.url = f"{server.rstrip('/')}/api/lots/{lot_id}/report"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client = client or httpx.Client(timeout=5.0, headers=headers)
        self.queue: deque[dict[str, Any]] = deque(maxlen=max_queue)
        self.sent = 0
        self.dropped = 0
        self._backoff_until = 0.0
        self._backoff = 1.0

    def publish(self, report: dict[str, Any]) -> None:
        if len(self.queue) == self.queue.maxlen:
            self.dropped += 1
        self.queue.append(report)
        self.flush()

    def flush(self) -> None:
        if time.monotonic() < self._backoff_until:
            return
        while self.queue:
            try:
                r = self.client.post(self.url, json=self.queue[0])
            except httpx.HTTPError as e:
                self._fail(f"server unreachable: {e}")
                return
            if r.status_code >= 500:
                self._fail(f"server error {r.status_code}")
                return
            if r.status_code >= 400:
                log.error("report rejected (%s): %s", r.status_code, r.text[:200])
            else:
                self.sent += 1
            self.queue.popleft()
            self._backoff = 1.0

    def _fail(self, why: str) -> None:
        log.warning("%s; %d reports queued, retrying in %.0fs", why, len(self.queue),
                    self._backoff)
        self._backoff_until = time.monotonic() + self._backoff
        self._backoff = min(self._backoff * 2, 60.0)


class Agent:
    def __init__(self, lot: Lot, classifier: OccupancyClassifier, publisher: Publisher,
                 filter_config: FilterConfig | None = None) -> None:
        self.lot = lot
        self.classifier = classifier
        self.publisher = publisher
        self.filter = BayFilter(filter_config)
        self.frames = 0
        self.infer_ms = 0.0

    def step(self, t: float, frame: Frame) -> dict[str, Any]:
        t0 = time.perf_counter()
        ids, patches = extract_patches(frame, self.lot, self.classifier.input_size)
        probs = self.classifier.predict_proba(patches)
        self.infer_ms = (time.perf_counter() - t0) * 1e3
        self.filter.update(t, dict(zip(ids, (float(p) for p in probs), strict=True)))
        report = {
            "t": t,
            "model": self.classifier.name,
            "bays": {b: {"occupied": s.occupied, "p": round(s.p, 3)}
                     for b, s in self.filter.states.items()},
        }
        self.publisher.publish(report)
        self.frames += 1
        return report

    def run(self, frames: Iterable[tuple[float, Frame]]) -> None:
        for t, frame in frames:
            self.step(t, frame)
            if self.frames % 20 == 0:
                free = sum(not s.occupied for s in self.filter.states.values())
                log.info("frame %d: %d/%d free, inference %.1f ms, sent %d, queued %d",
                         self.frames, free, self.lot.capacity, self.infer_ms,
                         self.publisher.sent, len(self.publisher.queue))


def run_agent(cfg: AgentConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    lot = Lot.load(cfg.lot_path)
    clf = load_classifier(cfg.model_path)
    source = open_source(cfg.source, fps=cfg.fps, speed=cfg.speed, loop=cfg.loop)
    sparse = getattr(source, "sparse", False)
    fcfg = FilterConfig(tau_s=1.0, min_dwell_s=0.0) if sparse else FilterConfig()
    agent = Agent(lot, clf, Publisher(cfg.server, lot.id, cfg.token, cfg.max_queue), fcfg)
    log.info("watching %s (%d bays) with %s", lot.name, lot.capacity, clf.name)
    agent.run(source)
