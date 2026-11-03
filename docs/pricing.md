# Pricing

## What the price is for

A car park can fail in two ways. If it is full, drivers circle the block looking for a
space, or give up. If it sits half empty at a price nobody will pay, the space is wasted.
Performance pricing, which San Francisco tried at scale with SFpark, treats price as the
lever between the two: raise it where a lot is always full, lower it where it is always
empty, and aim for the band where a driver arriving can nearly always find a space.

SPARK's rule (`spark.server.pricing.next_price`) is the same idea in one function. Each
review:

- occupancy above **85 %**: the price goes up by one step (0.25 per hour), or by up to two
  steps the further above the band it is;
- occupancy below **60 %**: the price goes down one step;
- in between: the price holds.

Prices are clamped to a floor and a ceiling (0.50 to 6.00 by default) and rounded to a
quarter. All of these are settings in `PricingConfig`.

## Four ways to apply the rule

| Policy | What it reacts to | How the current price is set |
|---|---|---|
| `fixed` | nothing | a constant base price |
| `responsive` | occupancy over the last hour | one rolling price, moved each hour |
| `predictive` | the forecast for the next hour (`spark.server.forecast`) | one rolling price, moved each hour |
| `banded` (default) | occupancy measured in that same hour, in previous weeks | a schedule: one price per day type (weekday, Saturday, Sunday) and hour; each hour, the band that just ended moves one step |

The forecast blends an exponentially weighted average of the same hour in previous weeks
with how far today is currently running above or below that average. The anomaly's
weight halves every hour of horizon.

## Simulation

`spark.sim` runs every policy against the same simulated demand, in 5-minute steps,
using the server's own `Pricer`, so what is simulated is what is deployed. The first 14
days let the forecast profile and the price bands learn and are not scored; the next 28
days are scored. Results are the mean and standard deviation over 5 random seeds.

    uv run spark sim --days 28 --seeds 5 --out results/pricing.json

### Assumptions

Every number is an assumption, chosen to be plausible and written down so it can be
argued with (`spark.sim.demand.DemandConfig`):

- a 60-space car park;
- Poisson arrivals: on weekdays a morning peak of up to 70 drivers an hour at 9:00, a lunch
  bump and an evening tail; weekends flatter, peaking at 28 an hour around noon;
- each driver's willingness to pay is log-normal, median 3.00 an hour; their planned stay
  is log-normal, median 75 minutes;
- a driver who finds the price above what they will pay goes elsewhere ("priced out"); a
  driver who would pay but finds the lot full leaves too ("turned away").

This is a toy model. It is useful for comparing policies with each other under identical
demand. It says nothing reliable about the revenue of a real site.

### Results

| Policy | Drivers served / day | Turned away (lot full) / day | Priced out / day | Time at 95%+ full | Mean occupancy | Mean price | Revenue / day |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed, base price (2.00) | 320 ± 2 | 14.6 ± 1.8 | 112 | 4.4% | 36% | 2.00 | 1016 |
| fixed, high price (4.00) | 141 ± 1 | 0.0 ± 0.0 | 306 | 0.0% | 16% | 4.00 | 894 |
| responsive (hourly) | 368 ± 1 | 41.3 ± 2.3 | 37 | 7.4% | 42% | 0.91 | 622 |
| predictive (hourly) | 381 ± 2 | 37.9 ± 3.1 | 28 | 8.0% | 43% | 0.79 | 596 |
| **banded by hour and day type** | **354 ± 1** | **3.9 ± 1.1** | 89 | **2.4%** | 40% | 1.09 | 761 |

### What the numbers say

**A single rolling price makes things worse.** The `responsive` policy turns away almost
three times as many drivers as a fixed price. Overnight and at weekends the lot is
nearly empty, so the price walks down to the floor. When the weekday morning peak
arrives, the price is at its cheapest and can only climb one step an hour, so the lot fills
before the price catches up. The `predictive` policy sees the peak coming but has the same
problem: one step an hour from the floor is not fast enough. This failure was not designed
in: it came out of the first simulation run, and it is the reason `banded` exists.

**Pricing each hour separately fixes it.** Under `banded`, the 9 a.m. weekday price learns
from earlier weekday 9 a.m.s and stays high, while the 3 a.m. price can sit at the floor
without dragging the morning down. Compared with the fixed base price it turns away
**73 % fewer drivers** (3.9 against 14.6 a day), serves **10 % more** (354 against 320),
and is at 95 % full or more about half as often.

**It is not a revenue policy.** Banded pricing earns about 25 % less than the fixed base
price in this model, because most hours are off-peak and their prices fall. A fixed high
price never fills up and still earns less than the fixed base price, because it prices
out more than half of demand. If an operator's goal is revenue, these targets are the
wrong ones; the target band and floor are settings, and the simulator is there to test
other choices.

**It needs time to learn.** With one week of history the bands have each seen only a few
updates and are not yet better than a fixed price on time spent full. After two weeks they
are clearly better. A new site should start from a sensible schedule, not from the
base price everywhere.

## In the server

The server runs a price review whenever a report arrives at least `review_every_s` (one
hour) of *observation* time after the previous review, so replays at 600x speed price
exactly as a live camera would. Each review is logged with its reason, for example
`weekday-09 band; weekday-08 band occupancy 93% above 85%`, and the band schedule is
stored in `price_bands`, so a restart keeps what was learned. Operators can trigger a
review from the dashboard (`POST /api/lots/{id}/prices/review`, operator token).
