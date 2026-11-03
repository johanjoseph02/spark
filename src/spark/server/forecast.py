"""Short-horizon occupancy forecast.

Car parks are strongly weekly-periodic, so the forecast for an hour is an exponentially
weighted average of the same hour on previous weeks (``HourOfWeekProfile``), corrected by
how far today is running above or below that profile. The correction fades with the
horizon: what is happening now says a lot about the next half hour and little about
tomorrow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

WEEK_HOURS = 7 * 24


def hour_of_week(t: float) -> int:
    """``t`` is seconds since the Unix epoch in lot-local time."""
    days = math.floor(t / 86400.0)
    weekday = (days + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    return int(weekday * 24 + math.floor((t % 86400.0) / 3600.0))


@dataclass
class HourOfWeekProfile:
    decay: float = 0.3  # weight of the newest week in each hour's average
    mean: list[float | None] = field(default_factory=lambda: [None] * WEEK_HOURS)

    def observe(self, t: float, occupancy: float) -> None:
        h = hour_of_week(t)
        m = self.mean[h]
        self.mean[h] = occupancy if m is None else m + self.decay * (occupancy - m)

    def expected(self, t: float) -> float | None:
        h = hour_of_week(t)
        if self.mean[h] is not None:
            return self.mean[h]
        # Fall back to the same hour on any other day, then to anything at all.
        same_hour = [m for i, m in enumerate(self.mean) if m is not None and i % 24 == h % 24]
        if same_hour:
            return sum(same_hour) / len(same_hour)
        seen = [m for m in self.mean if m is not None]
        return sum(seen) / len(seen) if seen else None


def forecast(profile: HourOfWeekProfile, now: float, current: float, horizon_s: float,
             half_life_s: float = 3600.0) -> float:
    """Occupancy expected at ``now + horizon_s``."""
    base_now = profile.expected(now)
    base_then = profile.expected(now + horizon_s)
    if base_then is None:
        return current
    anomaly = 0.0 if base_now is None else current - base_now
    keep: float = 0.5 ** (horizon_s / half_life_s)
    return float(min(max(base_then + keep * anomaly, 0.0), 1.0))
