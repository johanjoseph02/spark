from __future__ import annotations

import calendar
from dataclasses import replace
from pathlib import Path

import pytest

from spark.server.forecast import HourOfWeekProfile, forecast, hour_of_week
from spark.server.pricing import Pricer, PricingConfig, band, next_price
from spark.server.transit import haversine_m, load_gtfs, nearby
from spark.sim.demand import DemandConfig
from spark.sim.evaluate import compare_policies, simulate


def test_price_rises_when_full_falls_when_empty_holds_in_band() -> None:
    cfg = PricingConfig()
    assert next_price(cfg, 2.0, 0.95).price > 2.0
    assert next_price(cfg, 2.0, 0.30).price == 1.75
    assert next_price(cfg, 2.0, 0.70).price == 2.0


def test_price_is_bounded_and_rounded() -> None:
    cfg = PricingConfig(max_price=3.0, min_price=1.0)
    assert next_price(cfg, 3.0, 1.0).price == 3.0
    assert next_price(cfg, 1.0, 0.0).price == 1.0
    assert next_price(cfg, 2.0, 0.99).price * 4 == int(next_price(cfg, 2.0, 0.99).price * 4)


def test_bigger_overshoot_bigger_step() -> None:
    cfg = PricingConfig()
    assert next_price(cfg, 2.0, 1.0).price > next_price(cfg, 2.0, 0.88).price


def test_fixed_mode_and_validation() -> None:
    assert next_price(PricingConfig(mode="fixed"), 9.0, 1.0).price == 2.0
    with pytest.raises(ValueError):
        PricingConfig(target_low=0.9, target_high=0.5)
    with pytest.raises(ValueError):
        PricingConfig(mode="auction")


def test_bands_and_pricer() -> None:
    monday_9 = calendar.timegm((2024, 1, 1, 9, 30, 0))
    assert band(monday_9) == "weekday-09"
    assert band(monday_9 + 5 * 86400) == "sat-09"
    assert band(monday_9 + 6 * 86400) == "sun-09"
    p = Pricer(PricingConfig(mode="banded"))
    d = p.review(calendar.timegm((2024, 1, 1, 10, 0, 0)), measured=0.95, current=0.95)
    assert p.bands["weekday-09"] > 2.0  # the hour that ended moved
    assert d.price == 2.0  # 10:00 has no history yet
    r = Pricer(PricingConfig(mode="responsive"))
    assert r.review(0.0, 0.2, 0.2).price == 1.75


def test_hour_of_week() -> None:
    monday_9 = calendar.timegm((2024, 1, 1, 9, 30, 0))  # 1 Jan 2024 was a Monday
    assert hour_of_week(monday_9) == 9
    assert hour_of_week(monday_9 + 6 * 86400) == 6 * 24 + 9


def test_forecast_uses_weekly_profile_and_current_anomaly() -> None:
    prof = HourOfWeekProfile(decay=1.0)
    t0 = calendar.timegm((2024, 1, 1, 0, 0, 0))
    for h in range(24):
        prof.observe(t0 + h * 3600, 0.9 if 8 <= h < 18 else 0.1)
    now = t0 + 7 * 3600  # 07:00, the profile says 0.1
    assert forecast(prof, now, 0.1, 3600) == pytest.approx(0.9)  # 08:00 is busy
    # running 20 points above normal right now: part of that carries forward
    assert forecast(prof, now, 0.3, 3600) == pytest.approx(0.9 + 0.2 * 0.5)
    assert forecast(HourOfWeekProfile(), now, 0.42, 3600) == 0.42  # no history


def test_gtfs_nearby(tmp_path: Path) -> None:
    (tmp_path / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon,location_type\n"
        "s1,Near,43.7200,10.4240,0\ns2,Far,43.8000,10.5000,0\nst,Station,43.7201,10.4241,1\n")
    (tmp_path / "routes.txt").write_text("route_id,route_short_name\nr1,LAM Rossa\n")
    (tmp_path / "trips.txt").write_text("route_id,service_id,trip_id\nr1,x,t1\n")
    (tmp_path / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\nt1,08:00:00,08:00:00,s1,1\n")
    stops = load_gtfs(tmp_path)
    assert [s.id for s in stops] == ["s1", "s2"]  # the parent station is skipped
    found = nearby(stops, 43.7193, 10.4236)
    assert [f["id"] for f in found] == ["s1"]
    assert found[0]["routes"] == ["LAM Rossa"]
    assert 50 < float(str(found[0]["distance_m"])) < 120
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)


def test_simulation_is_deterministic_and_pricing_relieves_a_full_lot() -> None:
    demand = DemandConfig()
    base = PricingConfig()
    fixed = simulate(replace(base, mode="fixed"), demand, days=7, seed=3)
    again = simulate(replace(base, mode="fixed"), demand, days=7, seed=3)
    assert fixed == again
    banded = simulate(replace(base, mode="banded"), demand, days=7, seed=3)
    assert banded.share_time_full < fixed.share_time_full
    assert banded.turned_away_per_day < fixed.turned_away_per_day
    assert banded.served_per_day > fixed.served_per_day


def test_compare_policies_shape() -> None:
    out = compare_policies(days=2, seeds=1)
    assert set(out["policies"]) == {"fixed, base price", "fixed, high price",
                                    "responsive (hourly)", "predictive (hourly)",
                                    "banded by hour and day type"}
    assert "revenue_per_day" in out["policies"]["responsive (hourly)"]["mean"]
