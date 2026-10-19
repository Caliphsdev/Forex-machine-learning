"""Tests for the feature builder."""
import numpy as np
import pandas as pd
import pytest

from fxml.features.builder import FEATURE_COLUMNS, build_features


def make_bars(n: int = 80, scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-01 00:00", periods=n, freq="h", tz="UTC", name="time")
    close = 1.1 + np.cumsum(rng.normal(0, 0.001, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0005, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0005, n))
    return pd.DataFrame(
        {
            "open": open_ * scale,
            "high": high * scale,
            "low": low * scale,
            "close": close * scale,
            "tick_volume": rng.integers(100, 1000, n),
            "spread": 7,
            "real_volume": 0,
        },
        index=index,
    )


def test_columns_match_the_declared_list_and_index_is_kept():
    bars = make_bars()
    features = build_features(bars)
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.index.equals(bars.index)


def test_features_do_not_look_into_the_future():
    bars = make_bars()
    full = build_features(bars)
    truncated = build_features(bars.iloc[:50])
    pd.testing.assert_frame_equal(full.iloc[:50], truncated)


def test_price_based_features_are_scale_free():
    small = build_features(make_bars(scale=1.0))
    big = build_features(make_bars(scale=100.0))
    cols = ["ret_1", "ret_4", "ret_24", "log_atr", "log_range", "log_vol_24"]
    pd.testing.assert_frame_equal(small[cols], big[cols])


def test_time_features_use_the_utc_hour():
    features = build_features(make_bars())
    assert features["hour_sin"].iloc[0] == pytest.approx(0.0)   # 00:00
    assert features["hour_cos"].iloc[0] == pytest.approx(1.0)
    assert features["hour_sin"].iloc[6] == pytest.approx(1.0)   # 06:00
    assert features["hour_cos"].iloc[6] == pytest.approx(0.0, abs=1e-12)
