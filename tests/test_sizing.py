"""Tests for risk-based position sizing on daily bars."""
import numpy as np
import pandas as pd
import pytest

from fxml.features.sizing import (TRADING_DAYS, daily_vol, risk_scaled_position,
                                  vol_target_multiplier)


def alternating_close(move: float, n: int = 200) -> pd.Series:
    """A close series that moves +move, -move, +move... in log terms: constant daily volatility."""
    index = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC", name="time")
    steps = np.where(np.arange(n) % 2 == 0, move, -move)
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=index)


def test_daily_vol_of_constant_moves_is_that_move():
    vol = daily_vol(alternating_close(0.01))
    assert vol.dropna().to_numpy() == pytest.approx(0.01)


def test_daily_vol_needs_a_full_span_of_returns_first():
    vol = daily_vol(alternating_close(0.01))
    assert vol.iloc[:60].isna().all()
    assert vol.iloc[60:].notna().all()


def test_position_is_the_risk_target_divided_by_annual_volatility():
    close = alternating_close(0.01)
    position = risk_scaled_position(pd.Series(1.0, index=close.index), close, target=0.10)
    expected = 0.10 / (0.01 * np.sqrt(TRADING_DAYS))
    assert position.dropna().to_numpy() == pytest.approx(expected)


def test_doubling_the_volatility_halves_the_position():
    calm, wild = alternating_close(0.005), alternating_close(0.010)
    ones = pd.Series(1.0, index=calm.index)
    ratio = risk_scaled_position(ones, wild).iloc[-1] / risk_scaled_position(ones, calm).iloc[-1]
    assert ratio == pytest.approx(0.5)


def test_position_keeps_the_sign_and_size_of_the_signal():
    close = alternating_close(0.01)
    full = risk_scaled_position(pd.Series(1.0, index=close.index), close).iloc[-1]
    short_third = risk_scaled_position(pd.Series(-1 / 3, index=close.index), close).iloc[-1]
    assert short_third == pytest.approx(-full / 3)


def test_rewriting_future_prices_cannot_change_earlier_positions():
    rng = np.random.default_rng(1)
    steps = rng.normal(0, 0.005, 400)
    changed = steps.copy()
    changed[250:] = rng.normal(0, 0.05, 150)  # a much wilder future
    index = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC", name="time")
    signal = pd.Series(1.0, index=index)
    original = pd.Series(100 * np.exp(np.cumsum(steps)), index=index)
    rewritten = pd.Series(100 * np.exp(np.cumsum(changed)), index=index)
    pd.testing.assert_series_equal(risk_scaled_position(signal, original).iloc[:250],
                                   risk_scaled_position(signal, rewritten).iloc[:250])

def alternating_returns(move: float, n: int = 200) -> pd.Series:
    index = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC", name="time")
    return pd.Series(np.where(np.arange(n) % 2 == 0, move, -move), index=index)


def test_multiplier_scales_a_volatile_portfolio_down_to_the_target():
    multiplier = vol_target_multiplier(alternating_returns(0.01), target=0.10)
    assert multiplier.dropna().to_numpy() == pytest.approx(0.10 / (0.01 * np.sqrt(TRADING_DAYS)))


def test_multiplier_needs_a_full_span_of_returns_first():
    multiplier = vol_target_multiplier(alternating_returns(0.01))
    assert multiplier.iloc[:59].isna().all()
    assert multiplier.iloc[59:].notna().all()


def test_halving_the_portfolio_volatility_doubles_the_multiplier():
    wild = vol_target_multiplier(alternating_returns(0.02)).iloc[-1]
    calm = vol_target_multiplier(alternating_returns(0.01)).iloc[-1]
    assert calm / wild == pytest.approx(2.0)


def test_rewriting_future_returns_cannot_change_earlier_multipliers():
    rng = np.random.default_rng(2)
    index = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC", name="time")
    returns = pd.Series(rng.normal(0, 0.005, 400), index=index)
    changed = returns.copy()
    changed.iloc[250:] = rng.normal(0, 0.05, 150)
    pd.testing.assert_series_equal(vol_target_multiplier(returns).iloc[:250],
                                   vol_target_multiplier(changed).iloc[:250])
