"""Tests for the cost-aware backtest."""
import numpy as np
import pandas as pd
import pytest

from fxml.evaluation.backtest import backtest, spread_cost, threshold_sweep

INDEX = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")


def preds(prob_up=(0.9, 0.55, 0.1, 0.5), actual=(0.01, 0.01, -0.02, 0.0)):
    return pd.DataFrame({"prob_up": prob_up, "actual_return": actual}, index=INDEX)


def flat_cost(value=0.001):
    return pd.Series(value, index=INDEX)


def test_position_follows_confident_signals_only():
    result = backtest(preds(), flat_cost(), threshold=0.1, horizon=4)
    assert result.trades["position"].tolist() == [1.0, 0.0, -1.0, 0.0]


def test_gross_is_position_times_outcome_and_cost_only_when_trading():
    trades = backtest(preds(), flat_cost(0.001), threshold=0.1, horizon=4).trades
    assert trades["gross"].tolist() == pytest.approx([0.01, 0.0, 0.02, 0.0])
    assert trades["cost"].tolist() == pytest.approx([0.001, 0.0, 0.001, 0.0])
    assert trades["net"].tolist() == pytest.approx([0.009, 0.0, 0.019, 0.0])


def test_capital_is_shared_across_the_holding_period():
    result = backtest(preds(), flat_cost(0.001), threshold=0.1, horizon=4)
    assert result.equity.iloc[-1] == pytest.approx((0.009 + 0.019) / 4)
    assert result.gross_equity.iloc[-1] == pytest.approx((0.01 + 0.02) / 4)


def test_stats_are_in_basis_points_per_trade():
    stats = backtest(preds(), flat_cost(0.001), threshold=0.1, horizon=4).stats
    assert stats["n_trades"] == 2
    assert stats["hit_rate"] == 1.0
    assert stats["gross_bps_per_trade"] == pytest.approx(150.0)
    assert stats["net_bps_per_trade"] == pytest.approx(140.0)
    assert stats["max_drawdown"] == pytest.approx(0.0)


def test_a_losing_trade_creates_a_drawdown():
    p = preds(prob_up=(0.9, 0.9, 0.5, 0.5), actual=(0.01, -0.01, 0.0, 0.0))
    stats = backtest(p, flat_cost(0.0), threshold=0.1, horizon=1).stats
    assert stats["max_drawdown"] == pytest.approx(-0.01)


def test_extra_cost_is_charged_per_trade():
    base = backtest(preds(), flat_cost(0.0), threshold=0.1, horizon=4).stats
    slipped = backtest(preds(), flat_cost(0.0), threshold=0.1, horizon=4,
                       extra_cost_bps=1.0).stats
    assert base["net_bps_per_trade"] - slipped["net_bps_per_trade"] == pytest.approx(1.0)


def test_no_trades_gives_nan_per_trade_stats():
    stats = backtest(preds(), flat_cost(), threshold=0.9, horizon=4).stats
    assert stats["n_trades"] == 0
    assert np.isnan(stats["net_bps_per_trade"])


def test_cost_is_aligned_by_timestamp_and_missing_cost_raises():
    shuffled = flat_cost().iloc[::-1]
    assert backtest(preds(), shuffled, threshold=0.1, horizon=4).stats["n_trades"] == 2
    with pytest.raises(ValueError):
        backtest(preds(), flat_cost().iloc[:2], threshold=0.1, horizon=4)


def test_spread_cost_converts_points_to_a_return_fraction():
    bars = pd.DataFrame({"spread": [10, 20], "close": [1.0, 2.0]}, index=INDEX[:2])
    assert spread_cost(bars, point=1e-5).tolist() == pytest.approx([1e-4, 1e-4])


def test_sweep_has_a_row_per_threshold_and_trades_fall_as_threshold_rises():
    sweep = threshold_sweep(preds(), flat_cost(), horizon=4, thresholds=[0.0, 0.1, 0.45])
    assert sweep.index.tolist() == [0.0, 0.1, 0.45]
    assert sweep["n_trades"].is_monotonic_decreasing


def test_sharpe_annualises_mean_over_std_of_per_bar_returns():
    two = pd.DataFrame({"prob_up": [0.9, 0.9], "actual_return": [0.02, 0.0]}, index=INDEX[:2])
    stats = backtest(two, pd.Series(0.0, index=INDEX[:2]), threshold=0.1, horizon=1,
                     periods_per_year=4).stats
    # per-bar returns [0.02, 0.0]: mean 0.01, std 0.014142 -> 0.7071 * sqrt(4)
    assert stats["sharpe"] == pytest.approx(1.4142, abs=1e-3)


def test_sharpe_is_nan_when_there_are_no_trades():
    stats = backtest(preds(), flat_cost(), threshold=0.9, horizon=4).stats
    assert np.isnan(stats["sharpe"])
