"""Tests for running one walk-forward fold end to end (tiny CPU problem)."""
import pandas as pd
import pytest
import torch

from fxml.datasets.splits import split_validation, walk_forward_splits
from fxml.datasets.table import build_table
from fxml.training.experiment import Settings, run_fold, run_walk_forward
from tests.test_builder import make_bars

CPU = torch.device("cpu")
SETTINGS = Settings(lookback=8, horizon=4, hidden_size=8, val_size=60,
                    epochs=2, patience=2, batch_size=64)


@pytest.fixture(scope="module")
def table():
    return build_table(make_bars(700), horizon=SETTINGS.horizon)


def first_fold(table):
    return walk_forward_splits(len(table), train_size=300, test_size=60,
                               gap=SETTINGS.horizon)[0]


def test_predictions_cover_the_test_window_in_original_units(table):
    fold = first_fold(table)
    result = run_fold(table, fold, SETTINGS, device=CPU)
    assert len(result.predictions) == fold.test_end - fold.test_start
    assert result.predictions.index.equals(table.index[fold.test])
    assert list(result.predictions.columns) == [
        "actual_return", "pred_return", "prob_up", "actual_direction"]
    assert (result.predictions["actual_return"].to_numpy()
            == table["target_return"].iloc[fold.test].to_numpy()).all()


def test_metrics_are_present_and_sensible(table):
    result = run_fold(table, first_fold(table), SETTINGS, device=CPU)
    m = result.metrics
    assert 0.0 <= m["test_direction_accuracy"] <= 1.0
    assert 0.0 <= m["baseline_majority_accuracy"] <= 1.0
    assert m["test_rmse"] > 0 and m["baseline_zero_rmse"] > 0
    assert result.return_std > 0
    assert m["n_test"] == 60


def test_scaler_and_standardiser_use_only_the_training_rows(table):
    fold = first_fold(table)
    result = run_fold(table, fold, SETTINGS, device=CPU)
    (train_start, train_end), _ = split_validation(
        fold.train_start, fold.train_end, SETTINGS.val_size, SETTINGS.horizon)
    train_rows = table.iloc[train_start:train_end]
    assert result.scaler.center_["ret_1"] == pytest.approx(train_rows["ret_1"].median())
    assert result.return_std == pytest.approx(train_rows["target_return"].std())


def test_training_never_sees_the_test_window(table):
    fold = first_fold(table)
    altered = table.copy()
    altered.iloc[fold.test_start:] = altered.iloc[fold.test_start:] * 5.0  # rewrite the "future"
    original = run_fold(table, fold, SETTINGS, device=CPU)
    changed = run_fold(altered, fold, SETTINGS, device=CPU)
    pd.testing.assert_frame_equal(original.history, changed.history)
    pd.testing.assert_series_equal(original.scaler.center_, changed.scaler.center_)
    assert original.return_std == changed.return_std
    assert not original.predictions["pred_return"].equals(changed.predictions["pred_return"])

def test_walk_forward_collects_metrics_and_predictions_for_every_fold(table):
    folds = walk_forward_splits(len(table), 300, 60, gap=SETTINGS.horizon)[:2]
    result = run_walk_forward(table, folds, SETTINGS, device=CPU, verbose=False)
    assert result.metrics.index.tolist() == [1, 2]
    assert "test_direction_accuracy" in result.metrics.columns
    assert result.metrics["test_from"].is_monotonic_increasing
    assert set(result.predictions["fold"]) == {1, 2}
    assert len(result.predictions) == 120
    assert result.predictions.index.is_unique and result.predictions.index.is_monotonic_increasing
