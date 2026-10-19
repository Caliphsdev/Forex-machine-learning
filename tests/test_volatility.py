"""Tests for the volatility feature set and table."""
import pandas as pd

from fxml.datasets.table import VOL_TARGET, build_vol_table
from fxml.features.volatility import VOL_FEATURE_COLUMNS, build_vol_features
from tests.test_builder import make_bars


def test_vol_features_have_declared_columns_and_keep_the_index():
    bars = make_bars(700)
    features = build_vol_features(bars)
    assert list(features.columns) == VOL_FEATURE_COLUMNS
    assert features.index.equals(bars.index)


def test_vol_features_do_not_look_into_the_future():
    bars = make_bars(700)
    pd.testing.assert_frame_equal(build_vol_features(bars).iloc[:600],
                                  build_vol_features(bars.iloc[:600]))


def test_vol_price_features_are_scale_free():
    cols = ["log_rv_6", "log_rv_24", "log_rv_120", "log_rv_500", "log_range", "log_atr"]
    pd.testing.assert_frame_equal(build_vol_features(make_bars(700, scale=1.0))[cols],
                                  build_vol_features(make_bars(700, scale=100.0))[cols])


def test_vol_table_has_features_then_target_and_no_nans():
    table = build_vol_table(make_bars(700), horizon=24)
    assert list(table.columns) == VOL_FEATURE_COLUMNS + [VOL_TARGET]
    assert not table.isna().any().any()


def test_vol_table_drops_the_warmup_head_and_the_unlabelled_tail():
    bars = make_bars(700)
    table = build_vol_table(bars, horizon=24)
    assert len(table) == 700 - 500 - 24
    assert table.index[0] == bars.index[500]
    assert table.index[-1] == bars.index[675]
