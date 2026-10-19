"""Daily trading engine: turn end-of-day positions into profit and loss, with fill costs."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxml.features.sizing import PORTFOLIO_VOL_TARGET, risk_scaled_position, vol_target_multiplier
from fxml.features.trend import trend_signal

SIZINGS = ("vol_targeted", "equal_notional")


def net_daily_pnl(position: pd.Series, open_: pd.Series, spread_fraction: pd.Series) -> pd.DataFrame:
    """Daily log-return profit and loss for one pair.

    `position` is the exposure decided at each day's close. It is filled at the next day's open
    and held to the open after that, so a position never earns a return that started before the
    decision. Every change in position pays half the spread at the fill. A missing position
    counts as flat. The last day has no next open, so its gross and net are NaN.

    All three inputs share one daily index. `spread_fraction` is the full spread as a fraction of
    price at each day's open. Columns: held, gross, cost, net.
    """
    if spread_fraction.reindex(position.index).isna().any():
        raise ValueError("spread is missing for some days")
    held = position.shift(1).fillna(0.0)  # decided at yesterday's close, filled at today's open
    day_return = np.log(open_.shift(-1) / open_)  # today's open to tomorrow's open
    gross = held * day_return
    cost = held.diff().abs().fillna(0.0) * spread_fraction / 2
    return pd.DataFrame({"held": held, "gross": gross, "cost": cost, "net": gross - cost})


def fill_spread_fraction(daily: pd.DataFrame, point: float, recorded_from: pd.Timestamp,
                         cost_multiplier: float = 1.0) -> pd.Series:
    """Full spread at each day's fill bar as a fraction of the open price.

    From `recorded_from` the bar's recorded spread is used. Earlier spreads are placeholders, and
    a recorded spread of zero is a data gap, not a free trade; both take the median of the
    positive recorded spreads. `cost_multiplier` scales every cost (2.0 for the stress test).
    """
    recorded = (daily.index >= recorded_from) & (daily["open_spread"] > 0).to_numpy()
    typical = daily.loc[recorded, "open_spread"].median()
    points = daily["open_spread"].where(recorded, typical)
    return points * point / daily["open"] * cost_multiplier

@dataclass
class DailyPortfolio:
    daily: pd.DataFrame      # portfolio gross, cost, net log returns and gross exposure per day
    multiplier: pd.Series    # portfolio volatility multiplier (1.0 for equal notional)
    per_pair: pd.DataFrame   # net log return per pair


def _add_up(frames) -> pd.Series:
    """Sum aligned series across pairs; a day is missing only if every pair is missing."""
    return pd.concat(list(frames), axis=1).sum(axis=1, min_count=1)


def run_daily_portfolio(daily_by_pair: dict, points: dict, recorded_from: pd.Timestamp,
                        cost_multiplier: float = 1.0, sizing: str = "vol_targeted",
                        target: float = PORTFOLIO_VOL_TARGET) -> DailyPortfolio:
    """Trend-follow every pair on daily bars and combine them into one portfolio.

    `daily_by_pair` maps a symbol to its `to_daily` bars and `points` to its point size.
    sizing "vol_targeted" sizes each pair to equal risk, then scales the whole portfolio to
    `target` annual volatility using only past returns. "equal_notional" gives each pair
    signal / number of pairs with no volatility scaling (the comparison baseline).
    """
    if sizing not in SIZINGS:
        raise ValueError(f"sizing must be one of {SIZINGS}, got {sizing!r}")
    positions, fractions = {}, {}
    for name, bars in daily_by_pair.items():
        signal = trend_signal(bars["close"])
        positions[name] = (risk_scaled_position(signal, bars["close"]) if sizing == "vol_targeted"
                           else signal / len(daily_by_pair))
        fractions[name] = fill_spread_fraction(bars, points[name], recorded_from, cost_multiplier)

    if sizing == "vol_targeted":
        raw = {}
        for name, bars in daily_by_pair.items():
            free = pd.Series(0.0, index=bars.index)
            gross = net_daily_pnl(positions[name], bars["open"], free)["gross"]
            raw[name] = gross.where(positions[name].shift(1).notna())  # no position yet: no data
        multiplier = vol_target_multiplier(_add_up(raw.values()), target)
        multiplier = multiplier.replace([np.inf, -np.inf], np.nan)
    else:
        union = pd.concat(list(positions.values()), axis=1).index
        multiplier = pd.Series(1.0, index=union)

    pnl = {}
    for name, bars in daily_by_pair.items():
        scaled = positions[name] * multiplier.reindex(bars.index)
        pnl[name] = net_daily_pnl(scaled, bars["open"], fractions[name])
    daily = pd.DataFrame({
        "gross": _add_up(p["gross"] for p in pnl.values()),
        "cost": _add_up(p["cost"] for p in pnl.values()),
        "net": _add_up(p["net"] for p in pnl.values()),
        "exposure": _add_up(p["held"].abs() for p in pnl.values()),
    })
    per_pair = pd.DataFrame({name: p["net"] for name, p in pnl.items()})
    return DailyPortfolio(daily, multiplier, per_pair)
