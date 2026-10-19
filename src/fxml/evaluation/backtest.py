"""Simulate trading the predictions with spread and slippage costs."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

BPS = 1e4


@dataclass
class BacktestResult:
    trades: pd.DataFrame      # per bar: position, gross, cost, net (log-return units)
    equity: pd.Series         # cumulative net return
    gross_equity: pd.Series   # cumulative return before costs
    stats: dict


def spread_cost(bars: pd.DataFrame, point: float) -> pd.Series:
    """Round-trip spread as a fraction of price: spread_points * point / close."""
    return bars["spread"] * point / bars["close"]


def backtest(predictions: pd.DataFrame, cost: pd.Series, threshold: float, horizon: int,
             extra_cost_bps: float = 0.0, periods_per_year: float = 6200) -> BacktestResult:
    """Trade every bar where |P(up) - 0.5| > threshold, holding `horizon` bars.

    A new position opens each qualifying bar, so `horizon` positions overlap and
    each gets 1/horizon of the capital. Costs are paid once per trade.
    """
    edge = predictions["prob_up"] - 0.5
    position = pd.Series(np.sign(edge), index=predictions.index).where(edge.abs() > threshold, 0.0)
    cost = cost.reindex(predictions.index)
    if cost.isna().any():
        raise ValueError("cost is missing for some prediction timestamps")

    gross = position * predictions["actual_return"]
    trade_cost = position.abs() * (cost + extra_cost_bps / BPS)
    net = gross - trade_cost
    per_bar = net / horizon
    equity = per_bar.cumsum()
    bar_std = per_bar.std()
    gross_equity = (gross / horizon).cumsum()

    trades = pd.DataFrame({"position": position, "gross": gross, "cost": trade_cost, "net": net})
    active = trades[trades["position"] != 0]
    n = len(active)
    stats = {
        "threshold": threshold,
        "n_trades": n,
        "hit_rate": float((active["gross"] > 0).mean()) if n else np.nan,
        "gross_bps_per_trade": float(active["gross"].mean() * BPS) if n else np.nan,
        "net_bps_per_trade": float(active["net"].mean() * BPS) if n else np.nan,
        "sharpe": float(per_bar.mean() / bar_std * np.sqrt(periods_per_year)) if bar_std > 0 else np.nan,
        "total_return": float(equity.iloc[-1]),
        "max_drawdown": float((equity - equity.cummax()).min()),
    }
    return BacktestResult(trades, equity, gross_equity, stats)


def threshold_sweep(predictions: pd.DataFrame, cost: pd.Series, horizon: int, thresholds,
                    extra_cost_bps: float = 0.0) -> pd.DataFrame:
    """One row of stats per confidence threshold."""
    rows = [backtest(predictions, cost, t, horizon, extra_cost_bps).stats for t in thresholds]
    return pd.DataFrame(rows).set_index("threshold")
