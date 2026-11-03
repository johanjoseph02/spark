"""Run every pricing policy against the same simulated demand and compare outcomes.

    uv run spark sim --days 28 --seeds 5 --out results/pricing.json

The policies use the server's own ``next_price`` and ``forecast`` functions, so what is
simulated is what is deployed.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from spark.server.forecast import HourOfWeekProfile
from spark.server.pricing import Pricer, PricingConfig
from spark.sim.demand import DemandConfig, draw_arrivals

STEP_S = 300.0


@dataclass(frozen=True)
class Outcome:
    mean_occupancy: float
    share_time_full: float
    turned_away_per_day: float
    priced_out_per_day: float
    served_per_day: float
    revenue_per_day: float
    mean_price: float


def simulate(policy: PricingConfig, demand: DemandConfig, days: int, seed: int,
             warmup_days: int = 14) -> Outcome:
    """Discrete-time simulation in 5-minute steps. The first ``warmup_days`` let the
    forecast profile and the price bands learn, and are not scored."""
    rng = np.random.default_rng(seed)
    total_days = days + warmup_days
    departures: list[float] = []  # departure times of parked cars
    pricer = Pricer(policy)
    price = pricer.price
    profile = HourOfWeekProfile()
    window: list[float] = []
    last_review = 0.0
    occ_series, full_steps = [], 0
    turned = priced = served = 0
    revenue, price_sum, steps = 0.0, 0.0, 0
    hour_acc: list[float] = []

    for k in range(int(total_days * 86400 / STEP_S)):
        t = k * STEP_S
        scored = t >= warmup_days * 86400
        departures = [d for d in departures if d > t]
        wtp, stay = draw_arrivals(demand, t, STEP_S, rng)
        for w, s in zip(wtp, stay, strict=True):
            if price > w:
                priced += scored
            elif len(departures) >= demand.capacity:
                turned += scored
            else:
                departures.append(t + s)
                served += scored
                if scored:
                    revenue += price * s / 3600.0
        occ = len(departures) / demand.capacity
        window.append(occ)
        hour_acc.append(occ)
        if len(hour_acc) * STEP_S >= 3600:
            profile.observe(t, statistics.fmean(hour_acc))
            hour_acc = []
        if scored:
            occ_series.append(occ)
            full_steps += occ >= 0.95
            price_sum += price
            steps += 1
        if t - last_review >= policy.review_every_s:
            measured = statistics.fmean(window) if window else 0.0
            price = pricer.review(t, measured, occ, profile).price
            window = []
            last_review = t

    return Outcome(
        mean_occupancy=statistics.fmean(occ_series),
        share_time_full=full_steps / max(steps, 1),
        turned_away_per_day=turned / days,
        priced_out_per_day=priced / days,
        served_per_day=served / days,
        revenue_per_day=revenue / days,
        mean_price=price_sum / max(steps, 1),
    )


def policies() -> dict[str, PricingConfig]:
    base = PricingConfig(review_every_s=3600.0)
    return {
        "fixed, base price": replace(base, mode="fixed"),
        "fixed, high price": replace(base, mode="fixed", base_price=4.0, max_price=6.0),
        "responsive (hourly)": replace(base, mode="responsive"),
        "predictive (hourly)": replace(base, mode="predictive"),
        "banded by hour and day type": replace(base, mode="banded"),
    }


def compare_policies(days: int = 28, seeds: int = 5,
                     demand: DemandConfig | None = None) -> dict[str, Any]:
    demand = demand or DemandConfig()
    out: dict[str, Any] = {"demand": asdict(demand), "days": days, "seeds": seeds,
                           "policies": {}}
    for name, pol in policies().items():
        runs = [simulate(pol, demand, days, seed) for seed in range(seeds)]
        out["policies"][name] = {
            "config": asdict(pol),
            "mean": {k: round(statistics.fmean(getattr(r, k) for r in runs), 3)
                     for k in Outcome.__dataclass_fields__},
            "sd": {k: round(statistics.pstdev(getattr(r, k) for r in runs), 3)
                   for k in Outcome.__dataclass_fields__},
        }
    return out
