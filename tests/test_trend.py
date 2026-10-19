"""Tests for the daily trend signal."""
import numpy as np
import pandas as pd
import pytest

from fxml.features.trend import trend_signal


def close_from_log_steps(steps) -> pd.Series:
    """A daily close series whose log price moves by `steps` each day."""
    index = pd.date_range("2020-01-01", periods=len(steps), freq="D", tz="UTC", name="time")
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=index)


def test_a_steady_rise_gives_plus_one_and_a_steady_fall_gives_minus_one():
    up = trend_signal(close_from_log_steps(np.full(200, 0.01)))
    down = trend_signal(close_from_log_steps(np.full(200, -0.01)))
    assert (up.dropna() == 1.0).all()
    assert (down.dropna() == -1.0).all()


def test_signal_is_missing_until_the_longest_lookback_has_history():
    signal = trend_signal(close_from_log_steps(np.full(200, 0.01)))
    assert signal.iloc[:126].isna().all()
    assert signal.iloc[126:].notna().all()


def test_signal_averages_the_three_speeds():
    # rises for 150 days, then falls: the 21-day trend turns down, the 63 and 126-day stay up
    steps = np.r_[0.0, np.full(149, 0.01), np.full(21, -0.01)]  # 171 days, last index 170
    signal = trend_signal(close_from_log_steps(steps))
    assert signal.iloc[-1] == pytest.approx(1 / 3)


def test_rewriting_future_prices_cannot_change_earlier_signals():
    rng = np.random.default_rng(0)
    steps = rng.normal(0, 0.005, 400)
    original = close_from_log_steps(steps)
    changed_steps = steps.copy()
    changed_steps[250:] = rng.normal(0.05, 0.05, 150)  # a very different future
    rewritten = close_from_log_steps(changed_steps)
    pd.testing.assert_series_equal(trend_signal(original).iloc[:250],
                                   trend_signal(rewritten).iloc[:250])
