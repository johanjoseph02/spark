from __future__ import annotations

import pytest

from spark.vision.temporal import BayFilter, FilterConfig


def test_first_frame_sets_state() -> None:
    f = BayFilter()
    changes = f.update(0.0, {"a": 0.9, "b": 0.1})
    assert {c.bay_id: c.occupied for c in changes} == {"a": True, "b": False}


def test_single_frame_glitch_is_ignored() -> None:
    f = BayFilter(FilterConfig(tau_s=2.0, min_dwell_s=2.0))
    f.update(0.0, {"a": 0.05})
    t = 0.0
    for _ in range(10):
        t += 0.5
        f.update(t, {"a": 0.05})
    # a person walks across the bay for one frame
    assert f.update(t + 0.5, {"a": 0.99}) == []
    assert f.snapshot() == {"a": False}


def test_real_arrival_flips_after_dwell() -> None:
    cfg = FilterConfig(tau_s=1.0, min_dwell_s=2.0)
    f = BayFilter(cfg)
    f.update(0.0, {"a": 0.0})
    flipped_at = None
    t = 0.0
    while t < 20 and flipped_at is None:
        t += 0.5
        if f.update(t, {"a": 0.95}):
            flipped_at = t
    assert flipped_at is not None
    assert flipped_at >= cfg.min_dwell_s
    assert f.snapshot() == {"a": True}


def test_behaviour_does_not_depend_on_frame_rate() -> None:
    def flip_time(fps: float) -> float:
        f = BayFilter(FilterConfig(tau_s=2.0, min_dwell_s=1.0))
        f.update(0.0, {"a": 0.0})
        t = 0.0
        while True:
            t += 1.0 / fps
            if f.update(t, {"a": 1.0}):
                return t

    assert flip_time(2.0) == pytest.approx(flip_time(30.0), abs=0.6)


def test_hysteresis_band_holds_state() -> None:
    f = BayFilter(FilterConfig(tau_s=0.1, min_dwell_s=0.0))
    f.update(0.0, {"a": 0.9})
    for i in range(1, 20):
        assert f.update(float(i), {"a": 0.5}) == []  # inside the band: no change


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        FilterConfig(on_threshold=0.4, off_threshold=0.6)
    with pytest.raises(ValueError):
        FilterConfig(tau_s=0)
