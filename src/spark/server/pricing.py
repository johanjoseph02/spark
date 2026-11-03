"""Demand-responsive pricing.

Price is a tool for keeping a lot *usefully* full: too full and drivers circle looking
for a space, too empty and the space is wasted. Each review, a price moves one step
toward the band where a driver arriving can almost always find a space.

Four policies, all sharing the same step rule (``next_price``):

* ``fixed``: one price, never changes. The baseline.
* ``responsive``: one rolling price, reviewed every period on the occupancy just measured.
  Simple, and it lags: overnight it drifts to the floor, so it meets the morning peak at
  its cheapest. The simulation in docs/pricing.md shows it doing worse than a fixed price.
* ``predictive``: one rolling price, reviewed on the occupancy *forecast* for the next
  period. It sees the peak coming but still climbs one step at a time.
* ``banded``: a separate price for each day type (weekday, Saturday, Sunday) and hour, the
  approach SFpark took with time-of-day rates. After each hour, that hour's band moves one
  step on what was measured in it, so next week's 9 a.m. price learns from this week's
  9 a.m. occupancy. The current price is simply the band for the current hour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from spark.server.forecast import HourOfWeekProfile, forecast

MODES = ("fixed", "responsive", "predictive", "banded")


@dataclass(frozen=True)
class PricingConfig:
    target_low: float = 0.60
    target_high: float = 0.85
    step: float = 0.25  # currency units per hour, per review
    min_price: float = 0.50
    max_price: float = 6.00
    base_price: float = 2.00
    review_every_s: float = 3600.0
    mode: str = "banded"

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_low < self.target_high <= 1.0:
            raise ValueError("need 0 <= target_low < target_high <= 1")
        if not 0 <= self.min_price <= self.base_price <= self.max_price:
            raise ValueError("need min_price <= base_price <= max_price")
        if self.mode not in MODES:
            raise ValueError(f"unknown pricing mode {self.mode!r}")


@dataclass(frozen=True)
class PriceDecision:
    price: float
    previous: float
    occupancy: float
    reason: str


def next_price(cfg: PricingConfig, previous: float, occupancy: float) -> PriceDecision:
    """One step of the rule. ``occupancy`` is measured or forecast, in [0, 1]."""
    occ = min(max(occupancy, 0.0), 1.0)
    if cfg.mode == "fixed":
        return PriceDecision(cfg.base_price, previous, occ, "fixed price")
    if occ > cfg.target_high:
        # Scale the step with how far above the band the lot is, up to two steps.
        over = (occ - cfg.target_high) / max(1.0 - cfg.target_high, 1e-9)
        price = previous + cfg.step * (1.0 + min(over, 1.0))
        reason = f"occupancy {occ:.0%} above {cfg.target_high:.0%}"
    elif occ < cfg.target_low:
        price = previous - cfg.step
        reason = f"occupancy {occ:.0%} below {cfg.target_low:.0%}"
    else:
        price = previous
        reason = f"occupancy {occ:.0%} within target"
    price = round(min(max(price, cfg.min_price), cfg.max_price) * 4) / 4  # quarter units
    return PriceDecision(price, previous, occ, reason)


def band(t: float) -> str:
    """Day type and hour of lot-local time ``t``: ``weekday-09``, ``sat-14``, ``sun-00``."""
    days = math.floor(t / 86400.0)
    weekday = (days + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    kind = "weekday" if weekday < 5 else ("sat" if weekday == 5 else "sun")
    return f"{kind}-{int((t % 86400.0) // 3600):02d}"


@dataclass
class Pricer:
    """Stateful price setter for one lot. ``bands`` is persisted by the server."""

    cfg: PricingConfig
    price: float = -1.0
    bands: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.price < 0:
            self.price = self.cfg.base_price

    def review(self, now: float, measured: float, current: float,
               profile: HourOfWeekProfile | None = None) -> PriceDecision:
        """``measured``: mean occupancy over the period that just ended.
        ``current``: occupancy right now."""
        cfg = self.cfg
        previous = self.price
        if cfg.mode == "banded":
            ended = band(now - cfg.review_every_s / 2)
            step = next_price(cfg, self.bands.get(ended, cfg.base_price), measured)
            self.bands[ended] = step.price
            self.price = self.bands.get(band(now), cfg.base_price)
            reason = f"{band(now)} band; {ended} band {step.reason}"
            return PriceDecision(self.price, previous, measured, reason)
        basis = measured
        if cfg.mode == "predictive" and profile is not None:
            basis = forecast(profile, now, current, cfg.review_every_s)
        d = next_price(cfg, previous, basis)
        self.price = d.price
        reason = f"forecast {d.reason}" if cfg.mode == "predictive" else d.reason
        return PriceDecision(d.price, previous, basis, reason)
