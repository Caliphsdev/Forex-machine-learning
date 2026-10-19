"""One pre-registered run of the daily trend strategy: trade, score, judge, log."""
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from fxml.db.repository import METRIC_COLUMNS, load_bars, load_runs, save_run
from fxml.evaluation.bootstrap import block_bootstrap_sharpe
from fxml.evaluation.daily_backtest import run_daily_portfolio
from fxml.evaluation.portfolio import portfolio_stats
from fxml.evaluation.preregistered import confirmation_verdict, development_verdict
from fxml.features.resample import to_daily
from fxml.features.sizing import (PAIR_VOL_TARGET, PORTFOLIO_VOL_TARGET, TRADING_DAYS,
                                  VOL_SPAN)
from fxml.features.trend import TREND_LOOKBACKS

PERIODS = {"dev": ("2001-01-01", "2015-01-01"), "confirm": ("2015-01-01", None)}  # end exclusive
RUN_NAMES = {"dev": "daily-trend-dev", "confirm": "daily-trend-confirm"}
RECORDED_FROM = pd.Timestamp("2015-01-01", tz="UTC")
STRESS_MULTIPLIER = 2.0
MIN_FINAL_DAY_BARS = 20  # a final day with fewer H1 bars than this is unfinished


def complete_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop the final day if it is unfinished. Thin days earlier in the data are kept."""
    return daily.iloc[:-1] if daily["n_bars"].iloc[-1] < MIN_FINAL_DAY_BARS else daily


def load_daily_pairs(conn, symbols) -> dict:
    """Daily bars for each symbol, built from its stored H1 bars."""
    return {s: complete_days(to_daily(load_bars(conn, s, "H1"))) for s in symbols}


def provenance(project_root: Path, preregistration: Path) -> dict:
    """Which code and which pass marks a run used, for the run log."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=project_root, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "preregistration_sha256": hashlib.sha256(Path(preregistration).read_bytes()).hexdigest(),
    }


def in_period(series: pd.Series, period: str) -> pd.Series:
    """The part of a daily series inside the period, without missing days."""
    start, end = PERIODS[period]
    keep = series.index >= pd.Timestamp(start, tz="UTC")
    if end is not None:
        keep &= series.index < pd.Timestamp(end, tz="UTC")
    return series[keep].dropna()


def summarise(returns: pd.Series, resamples: int = 2000) -> dict:
    """Sharpe with a 90% bootstrap interval, return, volatility and drawdown of daily returns."""
    returns = returns.dropna()
    stats = portfolio_stats(returns, TRADING_DAYS)
    low, high = block_bootstrap_sharpe(returns, TRADING_DAYS, n_resamples=resamples)
    return {
        "n_days": int(stats["n_bars"]),
        "sharpe": stats["sharpe"], "sharpe_ci90_low": low, "sharpe_ci90_high": high,
        "annual_return": float(returns.mean() * TRADING_DAYS),
        "annual_vol": float(returns.std() * np.sqrt(TRADING_DAYS)),
        "total_return": stats["total_return"], "max_drawdown": stats["max_drawdown"],
    }


def _sharpe(returns: pd.Series) -> float:
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)) if returns.std() > 0 else np.nan


def run_period(conn, daily_by_pair: dict, points: dict, period: str, provenance: dict | None = None,
               notes: str = "", resamples: int = 2000) -> dict:
    """Run one pre-registered period, score it against the pass marks and log it.

    "dev" scores 2001-2014 and compares volatility targeting with equal notional. "confirm"
    scores 2015 onward with normal and doubled costs, and can only be run once.
    """
    if period not in PERIODS:
        raise ValueError(f"period must be one of {list(PERIODS)}, got {period!r}")
    name = RUN_NAMES[period]
    if period == "confirm" and (load_runs(conn)["name"] == name).any():
        raise RuntimeError("the confirmation run has already been done; a retry needs new data")

    normal = run_daily_portfolio(daily_by_pair, points, RECORDED_FROM)
    net = in_period(normal.daily["net"], period)
    main = {
        **summarise(net, resamples),
        "gross_sharpe": _sharpe(in_period(normal.daily["gross"], period)),
        "cost_per_year": float(in_period(normal.daily["cost"], period).mean() * TRADING_DAYS),
        "exposure_median": float(in_period(normal.daily["exposure"], period).median()),
        "exposure_max": float(in_period(normal.daily["exposure"], period).max()),
    }
    if period == "dev":
        baseline = run_daily_portfolio(daily_by_pair, points, RECORDED_FROM, sizing="equal_notional")
        other = summarise(in_period(baseline.daily["net"], period), resamples)
        verdict = development_verdict(main, other)
        summary = {"targeted": main, "baseline": other}
    else:
        stressed = run_daily_portfolio(daily_by_pair, points, RECORDED_FROM,
                                       cost_multiplier=STRESS_MULTIPLIER)
        other = summarise(in_period(stressed.daily["net"], period), resamples)
        verdict = confirmation_verdict(main, other)
        summary = {"normal": main, "stressed": other}
    summary["verdict"] = {key: bool(value) for key, value in verdict.items()}
    summary["yearly_net"] = {str(year): float(total)
                             for year, total in net.groupby(net.index.year).sum().items()}
    summary["per_pair_sharpe"] = {pair: _sharpe(in_period(normal.per_pair[pair], period))
                                  for pair in normal.per_pair.columns}

    start, end = PERIODS[period]
    settings = {
        "period": period, "start": start, "end": end, "symbols": sorted(daily_by_pair),
        "lookbacks": list(TREND_LOOKBACKS), "vol_span": VOL_SPAN,
        "pair_vol_target": PAIR_VOL_TARGET, "portfolio_vol_target": PORTFOLIO_VOL_TARGET,
        "trading_days": TRADING_DAYS, "recorded_from": RECORDED_FROM.isoformat(),
        "stress_multiplier": STRESS_MULTIPLIER, "bootstrap_resamples": resamples,
        **(provenance or {}),
    }
    no_folds = pd.DataFrame(columns=["test_from", "test_to", *METRIC_COLUMNS])
    run_id = save_run(conn, name, "PORTFOLIO", "D1", settings, no_folds, summary, notes)
    return {"run_id": run_id, **summary}
