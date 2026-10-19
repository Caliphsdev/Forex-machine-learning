"""Tests for indicator functions."""
"""Tests for indicator functions."""
import math
import numpy as np
import pandas as pd
import pytest

from fxml.features.indicators import atr, log_return, true_range, log_realised_vol


def test_log_return_first_value_is_nan_and_rest_are_log_ratios():
    close = pd.Series([100.0, 110.0, 99.0])
    result = log_return(close)
    assert math.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(math.log(110 / 100))
    assert result.iloc[2] == pytest.approx(math.log(99 / 110))


def test_true_range_counts_gaps_from_previous_close():
    high = pd.Series([10.0, 12.0, 11.0])
    low = pd.Series([8.0, 11.5, 10.0])
    close = pd.Series([9.0, 11.8, 10.5])
    result = true_range(high, low, close)
    # bar 0: no previous close, so just high-low = 2
    # bar 1: gap up from 9, so high - prev_close = 3 beats high-low = 0.5
    # bar 2: gap down from 11.8, so prev_close - low = 1.8 beats high-low = 1
    assert result.tolist() == pytest.approx([2.0, 3.0, 1.8])


def test_atr_is_nan_during_warmup_then_settles_on_constant_range():
    close = pd.Series([100.0] * 6)
    result = atr(close + 1, close - 1, close, period=3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == pytest.approx([2.0] * 4)


def test_indicators_do_not_look_into_the_future():
    close = pd.Series([100, 101, 99, 102, 98, 103, 97, 104, 96, 105], dtype=float)
    high, low = close + 1.5, close - 1.0
    full = atr(high, low, close, period=3)
    truncated = atr(high.iloc[:7], low.iloc[:7], close.iloc[:7], period=3)
    pd.testing.assert_series_equal(full.iloc[:7], truncated)
    full_ret = log_return(close)
    pd.testing.assert_series_equal(full_ret.iloc[:7], log_return(close.iloc[:7]))

def test_log_realised_vol_of_constant_returns_is_the_log_of_that_return():
    close = pd.Series(100 * np.exp(0.001 * np.arange(10)))
    result = log_realised_vol(close, window=3)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3:].tolist() == pytest.approx([math.log(0.001)] * 7)


def test_log_realised_vol_is_root_mean_square_not_plain_mean():
    steps = np.array([0.0, 0.01, -0.01, 0.01, -0.01, 0.01])
    close = pd.Series(100 * np.exp(np.cumsum(steps)))
    result = log_realised_vol(close, window=4)
    assert result.iloc[-1] == pytest.approx(math.log(0.01))  # mean of returns is ~0, RMS is 0.01


def test_log_realised_vol_does_not_look_into_the_future():
    close = pd.Series(np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 40))))
    pd.testing.assert_series_equal(log_realised_vol(close, 5).iloc[:25],
                                   log_realised_vol(close.iloc[:25], 5))
