"""Per-bay smoothing of noisy classifier output.

A single frame is easily fooled: a pedestrian crossing an empty bay, a car driving past,
a shadow. ``BayFilter`` keeps an exponentially weighted probability per bay and only
flips the reported state when that probability crosses a hysteresis band and stays there
for a minimum dwell time. The time constant is in seconds, so the filter behaves the same
at 2 fps on a small board as at 30 fps on a laptop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BayState:
    occupied: bool
    p: float
    since: float
    pending_since: float | None = None


@dataclass(frozen=True)
class FilterConfig:
    tau_s: float = 4.0
    on_threshold: float = 0.65
    off_threshold: float = 0.35
    min_dwell_s: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.off_threshold < self.on_threshold <= 1.0:
            raise ValueError("need 0 <= off_threshold < on_threshold <= 1")
        if self.tau_s <= 0 or self.min_dwell_s < 0:
            raise ValueError("tau_s must be positive and min_dwell_s non-negative")


@dataclass(frozen=True)
class Change:
    bay_id: str
    occupied: bool
    t: float
    p: float


class BayFilter:
    def __init__(self, config: FilterConfig | None = None) -> None:
        self.config = config or FilterConfig()
        self.states: dict[str, BayState] = {}
        self._last_t: float | None = None

    def update(self, t: float, probs: dict[str, float]) -> list[Change]:
        """Feed one frame's occupancy probabilities at time ``t`` (seconds).

        Returns the bays whose reported state changed on this frame.
        """
        cfg = self.config
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t
        alpha = 1.0 - math.exp(-dt / cfg.tau_s) if dt > 0 else 1.0
        changes: list[Change] = []
        for bay_id, p_raw in probs.items():
            p_raw = min(max(float(p_raw), 0.0), 1.0)
            st = self.states.get(bay_id)
            if st is None:
                occ = p_raw >= 0.5
                self.states[bay_id] = BayState(occ, p_raw, t)
                changes.append(Change(bay_id, occ, t, p_raw))
                continue
            st.p += alpha * (p_raw - st.p)
            wants_flip = (not st.occupied and st.p >= cfg.on_threshold) or (
                st.occupied and st.p <= cfg.off_threshold
            )
            if not wants_flip:
                st.pending_since = None
                continue
            if st.pending_since is None:
                st.pending_since = t
            if t - st.pending_since >= cfg.min_dwell_s:
                st.occupied = not st.occupied
                st.since = t
                st.pending_since = None
                changes.append(Change(bay_id, st.occupied, t, st.p))
        return changes

    def snapshot(self) -> dict[str, bool]:
        return {k: v.occupied for k, v in self.states.items()}
