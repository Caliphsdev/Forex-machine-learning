"""One call from the database to a logged experiment result."""
import sqlite3
from dataclasses import asdict, dataclass

import pandas as pd
import torch

from fxml import config
from fxml.datasets.splits import walk_forward_splits
from fxml.datasets.table import build_table
from fxml.db.repository import load_bars, save_run
from fxml.evaluation.backtest import BPS, BacktestResult, backtest, spread_cost
from fxml.training.experiment import Settings, run_walk_forward


@dataclass(frozen=True)
class ExperimentPlan:
    """What data to use and how to judge the result (the model's own knobs are Settings)."""
    name: str
    symbol: str = "EURUSD"
    timeframe: str = "H1"
    start: str = config.TRAIN_START
    end: str | None = None                 # exclusive; None = up to the latest bar
    train_size: int = 31_000               # rows in the first training span
    test_size: int = 3_100                 # rows per test window
    confidence_quantile: float = 0.8       # trade only the most confident 20%
    flat_cost_bps: float | None = None     # None = charge each bar's recorded spread
    extra_cost_bps: float = 0.3            # slippage stress test, reported separately
    notes: str = ""


@dataclass
class ExperimentOutcome:
    summary: dict               # what is stored in SQLite, plus run_id
    backtest: BacktestResult
    predictions: pd.DataFrame   # every out-of-sample bar
    per_bar: pd.Series          # net return per bar, capital shared across the horizon


def run_experiment_detailed(conn: sqlite3.Connection, plan: ExperimentPlan,
                            settings: Settings = Settings(), device: torch.device | None = None,
                            verbose: bool = True) -> ExperimentOutcome:
    """Load bars, run the walk-forward, backtest with costs, save the run; return everything."""
    bars = load_bars(conn, plan.symbol, plan.timeframe, start=plan.start, end=plan.end)
    if bars.empty:
        raise ValueError(f"No {plan.symbol} {plan.timeframe} bars from {plan.start} to {plan.end}")

    table = build_table(bars, horizon=settings.horizon)
    folds = walk_forward_splits(len(table), plan.train_size, plan.test_size, gap=settings.horizon)
    wf = run_walk_forward(table, folds, settings, device, verbose)

    predictions = wf.predictions
    threshold = (predictions["prob_up"] - 0.5).abs().quantile(plan.confidence_quantile)
    if plan.flat_cost_bps is None:
        if plan.symbol not in config.POINT_SIZES:
            raise ValueError(f"Add {plan.symbol} to config.POINT_SIZES (1 point in price units)")
        cost = spread_cost(bars, config.POINT_SIZES[plan.symbol])
    else:
        cost = pd.Series(plan.flat_cost_bps / BPS, index=predictions.index)
    result = backtest(predictions, cost, threshold, settings.horizon)
    stressed = backtest(predictions, cost, threshold, settings.horizon,
                        extra_cost_bps=plan.extra_cost_bps)

    metrics = wf.metrics
    summary = {
        **result.stats,
        f"net_bps_with_{plan.extra_cost_bps:g}_slippage": stressed.stats["net_bps_per_trade"],
        "mean_accuracy": metrics["test_direction_accuracy"].mean(),
        "mean_rank_corr": metrics["test_rank_corr"].mean(),
        "folds_positive_rank_corr": int((metrics["test_rank_corr"] > 0).sum()),
        "n_folds": len(folds),
    }
    run_id = save_run(conn, plan.name, plan.symbol, plan.timeframe,
                      {"settings": asdict(settings), "plan": asdict(plan)},
                      metrics, summary=summary, notes=plan.notes)
    per_bar = result.trades["net"] / settings.horizon
    return ExperimentOutcome({"run_id": run_id, **summary}, result, predictions, per_bar)


def run_experiment(conn: sqlite3.Connection, plan: ExperimentPlan,
                   settings: Settings = Settings(), device: torch.device | None = None,
                   verbose: bool = True) -> dict:
    """Like run_experiment_detailed, but returns only the summary dict."""
    return run_experiment_detailed(conn, plan, settings, device, verbose).summary
