"""Synthetic demand for one car park.

Every number here is an assumption, chosen to be plausible and stated so it can be
argued with. The model exists to compare pricing *policies* under the same demand, not
to predict revenue for a real site.

* Arrivals are Poisson with a weekday profile (morning peak, lunch bump, evening tail)
  and a lower, flatter weekend. At the base price the weekday peak asks for more spaces
  than the lot has.
* Each arriving driver has a willingness to pay per hour (log-normal) and a planned stay
  (log-normal, median 75 minutes).
* A driver who finds the price above their willingness to pay goes elsewhere ("priced
  out"). A driver willing to pay who finds the lot full circles and leaves ("turned away").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DemandConfig:
    capacity: int = 60
    weekday_peak_per_h: float = 70.0
    weekend_peak_per_h: float = 28.0
    wtp_median: float = 3.0  # per hour
    wtp_sigma: float = 0.6
    stay_median_min: float = 75.0
    stay_sigma: float = 0.7


def arrival_rate(cfg: DemandConfig, t_s: float) -> float:
    """Expected arrivals per hour at ``t_s`` seconds after a Monday 00:00."""
    day = int(t_s // 86400) % 7
    h = (t_s % 86400) / 3600.0

    def bump(mu: float, sd: float) -> float:
        return math.exp(-0.5 * ((h - mu) / sd) ** 2)

    if day < 5:
        shape = 0.08 + 1.0 * bump(9.0, 1.2) + 0.55 * bump(13.0, 1.0) + 0.35 * bump(17.5, 1.5)
        return cfg.weekday_peak_per_h * shape
    shape = 0.08 + 1.0 * bump(12.0, 2.5)
    return cfg.weekend_peak_per_h * shape


def draw_arrivals(cfg: DemandConfig, t_s: float, dt_s: float,
                  rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(willingness to pay per hour, stay in seconds) for each driver arriving in a step."""
    n = rng.poisson(arrival_rate(cfg, t_s) * dt_s / 3600.0)
    wtp = cfg.wtp_median * np.exp(cfg.wtp_sigma * rng.standard_normal(n))
    stay = cfg.stay_median_min * 60.0 * np.exp(cfg.stay_sigma * rng.standard_normal(n))
    return wtp, stay
