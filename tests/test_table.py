"""Tests for the modelling table (features + targets, aligned and NaN-free)."""
from fxml.datasets.table import TARGET_COLUMNS, build_table
from fxml.features.builder import FEATURE_COLUMNS
from tests.test_builder import make_bars


def test_table_has_features_then_targets_and_no_nans():
    table = build_table(make_bars(80), horizon=4)
    assert list(table.columns) == FEATURE_COLUMNS + TARGET_COLUMNS
    assert not table.isna().any().any()


def test_table_drops_the_warmup_head_and_the_unlabelled_tail():
    bars = make_bars(80)
    table = build_table(bars, horizon=4)
    assert len(table) == 80 - 24 - 4  # 24 warm-up rows, 4 rows with no future
    assert table.index[0] == bars.index[24]
    assert table.index[-1] == bars.index[75]
