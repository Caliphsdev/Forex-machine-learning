"""Walk-forward scoring of simple forecasters (no neural net)."""
import numpy as np
import pandas as pd

from fxml.datasets.splits import Fold
from fxml.datasets.table import VOL_TARGET
from fxml.evaluation.metrics import oos_r2, rmse
from fxml.models.linear import LinearModel


def _fold_row(number: int, train_target: pd.Series, test: pd.DataFrame, predicted,
              target: str) -> dict:
    actual = test[target]
    predicted = pd.Series(np.asarray(predicted, dtype=float), index=actual.index)
    return {
        "fold": number,
        "test_from": test.index[0],
        "test_to": test.index[-1],
        "r2": oos_r2(actual, predicted, benchmark=train_target.mean()),
        "rmse": rmse(actual, predicted),
        "rank_corr": float(actual.corr(predicted, method="spearman")),
        "n_test": len(test),
    }


def walk_forward_forecast(table: pd.DataFrame, folds: list[Fold], feature_columns: list[str],
                          target: str = VOL_TARGET) -> pd.DataFrame:
    """Fit a LinearModel on each fold's training rows; score it on that fold's test rows."""
    rows = []
    for number, fold in enumerate(folds, start=1):
        train, test = table.iloc[fold.train], table.iloc[fold.test]
        model = LinearModel().fit(train[feature_columns], train[target])
        rows.append(_fold_row(number, train[target], test,
                              model.predict(test[feature_columns]), target))
    return pd.DataFrame(rows).set_index("fold")


def walk_forward_persistence(table: pd.DataFrame, folds: list[Fold], column: str,
                             target: str = VOL_TARGET) -> pd.DataFrame:
    """No fitting: predict that the future equals the current value of `column`."""
    rows = []
    for number, fold in enumerate(folds, start=1):
        train, test = table.iloc[fold.train], table.iloc[fold.test]
        rows.append(_fold_row(number, train[target], test, test[column].to_numpy(), target))
    return pd.DataFrame(rows).set_index("fold")

def walk_forward_predict(table: pd.DataFrame, folds: list[Fold], feature_columns: list[str],
                         target: str = VOL_TARGET) -> pd.Series:
    """Out-of-sample forecasts: each fold's model predicts only its own test rows."""
    pieces = []
    for fold in folds:
        train, test = table.iloc[fold.train], table.iloc[fold.test]
        model = LinearModel().fit(train[feature_columns], train[target])
        pieces.append(pd.Series(model.predict(test[feature_columns]), index=test.index))
    return pd.concat(pieces).rename("forecast")
