"""Tests for the linear baseline and the walk-forward forecast scorers."""
import numpy as np
import pandas as pd
import pytest

from fxml.datasets.splits import walk_forward_splits
from fxml.models.linear import LinearModel
from fxml.training.baselines import walk_forward_forecast, walk_forward_persistence, walk_forward_predict

TARGET = "target_log_vol"


def make_table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    table = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=index)
    table[TARGET] = 2 * table["a"] - table["b"]
    return table


def test_linear_model_recovers_exact_coefficients():
    table = make_table()
    table[TARGET] = 1 + 2 * table["a"] - 3 * table["b"]
    model = LinearModel().fit(table[["a", "b"]], table[TARGET])
    assert model.coef_ == pytest.approx([1.0, 2.0, -3.0])
    assert model.predict(table[["a", "b"]]) == pytest.approx(table[TARGET].to_numpy())


def test_linear_model_matches_columns_by_name_not_position():
    table = make_table()
    model = LinearModel().fit(table[["a", "b"]], table[TARGET])
    assert model.predict(table[["b", "a"]]) == pytest.approx(model.predict(table[["a", "b"]]))


def test_linear_model_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LinearModel().predict(make_table()[["a", "b"]])


def test_forecast_scores_each_fold_on_its_own_test_rows():
    table = make_table()
    folds = walk_forward_splits(len(table), train_size=200, test_size=50, gap=4)
    result = walk_forward_forecast(table, folds, ["a", "b"])
    assert result.index.tolist() == list(range(1, len(folds) + 1))
    assert (result["r2"] > 0.999).all()  # exact relationship: only possible if rows line up
    assert (result["n_test"] == 50).all()
    assert result["test_from"].is_monotonic_increasing


def test_persistence_of_the_target_itself_scores_one():
    table = make_table()
    table["same"] = table[TARGET]
    folds = walk_forward_splits(len(table), train_size=200, test_size=50, gap=4)
    assert (walk_forward_persistence(table, folds, "same")["r2"] > 0.999).all()


def test_forecast_never_fits_on_the_test_window():
    table = make_table()
    fold = walk_forward_splits(len(table), train_size=200, test_size=50, gap=4)[0]
    table.loc[table.index[fold.test_start:], TARGET] *= -1  # the relationship flips in the future
    result = walk_forward_forecast(table, [fold], ["a", "b"])
    assert result["r2"].iloc[0] < -2  # a model fit on train only predicts the old relationship

def test_predict_series_covers_exactly_the_test_rows_of_every_fold():
    table = make_table()
    folds = walk_forward_splits(len(table), train_size=200, test_size=50, gap=4)
    forecast = walk_forward_predict(table, folds, ["a", "b"])
    expected = table.index[folds[0].test].append([table.index[f.test] for f in folds[1:]])
    assert forecast.index.equals(expected)
    assert forecast.index.is_unique and forecast.index.is_monotonic_increasing


def test_predict_series_matches_an_exact_relationship():
    table = make_table()
    folds = walk_forward_splits(len(table), train_size=200, test_size=50, gap=4)
    forecast = walk_forward_predict(table, folds, ["a", "b"])
    assert forecast.to_numpy() == pytest.approx(table.loc[forecast.index, TARGET].to_numpy())


def test_first_fold_forecast_does_not_depend_on_future_targets():
    table = make_table()
    fold = walk_forward_splits(len(table), train_size=200, test_size=50, gap=4)[0]
    changed = table.copy()
    changed.loc[changed.index[fold.test_start:], TARGET] *= -7.0
    pd.testing.assert_series_equal(walk_forward_predict(table, [fold], ["a", "b"]),
                                   walk_forward_predict(changed, [fold], ["a", "b"]))
