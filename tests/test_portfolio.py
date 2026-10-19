"""Tests for combining several strategies into one portfolio."""
import numpy as np
import pandas as pd
import pytest

from fxml.evaluation.portfolio import combine_equal_weight, portfolio_stats, return_correlations

INDEX = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")


def test_combine_uses_only_shared_timestamps_and_averages():
    a = pd.Series([0.02, 0.04, 0.06, 0.08], index=INDEX)
    b = pd.Series([0.00, 0.02, 0.04], index=INDEX[1:])
    combined = combine_equal_weight({"a": a, "b": b})
    assert combined.index.equals(INDEX[1:])
    assert combined.tolist() == pytest.approx([0.02, 0.04, 0.06])


def test_portfolio_stats_known_values():
    stats = portfolio_stats(pd.Series([0.02, 0.0]), periods_per_year=4)
    assert stats["sharpe"] == pytest.approx(1.4142, abs=1e-3)
    assert stats["total_return"] == pytest.approx(0.02)
    drawdown = portfolio_stats(pd.Series([0.01, -0.02]))["max_drawdown"]
    assert drawdown == pytest.approx(-0.02)


def test_correlations_are_one_for_identical_series():
    a = pd.Series([0.1, -0.2, 0.3, 0.0], index=INDEX)
    corr = return_correlations({"a": a, "b": a * 2})
    assert corr.loc["a", "b"] == pytest.approx(1.0)


def test_uncorrelated_strategies_diversify():
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=5000, freq="h", tz="UTC")
    a = pd.Series(rng.normal(0.1, 1.0, 5000), index=index)
    b = pd.Series(rng.normal(0.1, 1.0, 5000), index=index)
    single = portfolio_stats(a)["sharpe"]
    both = portfolio_stats(combine_equal_weight({"a": a, "b": b}))["sharpe"]
    assert both > 1.3 * single  # theory says about 1.41x for two uncorrelated strategies
