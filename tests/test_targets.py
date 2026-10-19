"""Tests for target (label) construction."""
import math
import numpy as np
import pandas as pd
import pytest

from fxml.labels.targets import direction, forward_return, forward_log_vol
from fxml.features.indicators import log_realised_vol

def test_forward_return_one_bar_looks_one_step_ahead():
    close = pd.Series([100.0, 110.0, 99.0])
    result = forward_return(close, horizon=1)
    assert result.iloc[0] == pytest.approx(math.log(110 / 100))
    assert result.iloc[1] == pytest.approx(math.log(99 / 110))
    assert math.isnan(result.iloc[2])  # no future bar to look at


def test_forward_return_multi_bar_horizon_leaves_nan_tail():
    close = pd.Series([100.0, 110.0, 99.0, 105.0])
    result = forward_return(close, horizon=2)
    assert result.iloc[0] == pytest.approx(math.log(99 / 100))
    assert result.iloc[1] == pytest.approx(math.log(105 / 110))
    assert result.iloc[2:].isna().all()


def test_direction_is_one_for_up_zero_otherwise_and_keeps_nan():
    fwd = pd.Series([0.01, -0.02, 0.0, float("nan")])
    result = direction(fwd)
    assert result.iloc[:3].tolist() == [1.0, 0.0, 0.0]
    assert math.isnan(result.iloc[3])


def test_forward_log_vol_averages_the_next_bars_squared_returns():
    steps = np.array([0.0, 0.01, -0.02, 0.03, 0.01])
    close = pd.Series(np.exp(np.cumsum(steps)))
    result = forward_log_vol(close, horizon=2)
    assert result.iloc[0] == pytest.approx(0.5 * math.log((0.01**2 + 0.02**2) / 2))
    assert result.iloc[1] == pytest.approx(0.5 * math.log((0.02**2 + 0.03**2) / 2))
    assert result.iloc[2] == pytest.approx(0.5 * math.log((0.03**2 + 0.01**2) / 2))
    assert result.iloc[3:].isna().all()


def test_forward_log_vol_is_trailing_vol_shifted_back_by_the_horizon():
    close = pd.Series(np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 30))))
    pd.testing.assert_series_equal(forward_log_vol(close, 3), log_realised_vol(close, 3).shift(-3))
