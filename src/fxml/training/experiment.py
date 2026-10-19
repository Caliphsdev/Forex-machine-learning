"""Ties data + features + model + trainer together for one walk-forward fold."""
from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn

from fxml import config
from fxml.datasets.scaling import FeatureScaler
from fxml.datasets.splits import Fold, split_validation
from fxml.datasets.windows import WindowDataset
from fxml.evaluation.metrics import (
    directional_accuracy, majority_baseline_accuracy, mae, rmse,
)
from fxml.features.builder import CYCLICAL_COLUMNS, FEATURE_COLUMNS
from fxml.models.registry import build_model
from fxml.training.trainer import predict, train_model

FEATURE_CLIP = 10.0  # scaled features are clipped to +-this so rare spikes cannot dominate


@dataclass(frozen=True)
class Settings:
    model_name: str = "lstm"
    lookback: int = config.LOOKBACK
    horizon: int = config.LABEL_HORIZON
    hidden_size: int = config.HIDDEN_SIZE
    dropout: float = config.DROPOUT
    val_size: int = 3000  # rows held out at the end of the training span for early stopping
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 8
    seed: int = 0


@dataclass
class FoldResult:
    fold: Fold
    history: pd.DataFrame
    best_epoch: int
    metrics: dict
    predictions: pd.DataFrame  # one row per test bar, original units
    model: nn.Module
    scaler: FeatureScaler
    return_std: float


def run_fold(table: pd.DataFrame, fold: Fold, settings: Settings = Settings(),
             device: torch.device | None = None) -> FoldResult:
    """Train on one fold and score it on that fold's test window.

    Everything learned from data (scaler, return standardiser, weights, the best
    epoch) comes from the fold's training span only.
    """
    (train_start, train_end), (val_start, val_end) = split_validation(
        fold.train_start, fold.train_end, settings.val_size, gap=settings.horizon
    )
    features = table[FEATURE_COLUMNS]

    scaler = FeatureScaler(passthrough=CYCLICAL_COLUMNS).fit(features.iloc[train_start:train_end])
    x = scaler.transform(features).clip(-FEATURE_CLIP, FEATURE_CLIP)

    return_std = float(table["target_return"].iloc[train_start:train_end].std())
    y_return = table["target_return"] / return_std
    y_direction = table["target_direction"]

    def make_ds(start: int, end: int) -> WindowDataset:
        return WindowDataset(x, y_return, y_direction, lookback=settings.lookback,
                             start=max(start, settings.lookback - 1), end=end)

    train_ds = make_ds(train_start, train_end)
    val_ds = make_ds(val_start, val_end)
    test_ds = make_ds(fold.test_start, fold.test_end)

    torch.manual_seed(settings.seed)
    model = build_model(settings.model_name, n_features=len(FEATURE_COLUMNS),
                        hidden_size=settings.hidden_size, dropout=settings.dropout)
    trained = train_model(model, train_ds, val_ds, epochs=settings.epochs,
                          batch_size=settings.batch_size, lr=settings.lr,
                          patience=settings.patience, device=device, seed=settings.seed)

    pred_standardised, prob_up = predict(model, test_ds, device=device)
    rows = slice(test_ds.start, test_ds.end)
    predictions = pd.DataFrame(
        {
            "actual_return": table["target_return"].iloc[rows].to_numpy(),
            "pred_return": pred_standardised * return_std,
            "prob_up": prob_up,
            "actual_direction": table["target_direction"].iloc[rows].to_numpy(),
        },
        index=table.index[rows],
    )

    actual, predicted = predictions["actual_return"], predictions["pred_return"]
    metrics = {
        "test_direction_accuracy": directional_accuracy(predictions["actual_direction"], prob_up),
        "baseline_majority_accuracy": majority_baseline_accuracy(
            table["target_direction"].iloc[train_start:train_end], predictions["actual_direction"]),
        "test_rmse": rmse(actual, predicted),
        "baseline_zero_rmse": rmse(actual, actual * 0.0),
        "test_mae": mae(actual, predicted),
        "test_rank_corr": float(actual.corr(predicted, method="spearman")),
        "best_epoch": trained.best_epoch,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
    }
    return FoldResult(fold, trained.history, trained.best_epoch, metrics, predictions,
                      model, scaler, return_std)

@dataclass
class WalkForwardResult:
    metrics: pd.DataFrame      # one row per fold
    predictions: pd.DataFrame  # every test bar from every fold, with a "fold" column


def run_walk_forward(table: pd.DataFrame, folds: list[Fold], settings: Settings = Settings(),
                     device: torch.device | None = None,
                     verbose: bool = True) -> WalkForwardResult:
    """Train and score every fold in order; collect metrics and out-of-sample predictions."""
    rows, predictions = [], []
    for number, fold in enumerate(folds, start=1):
        result = run_fold(table, fold, settings, device)
        frame = result.predictions
        rows.append({"fold": number, "test_from": frame.index[0], "test_to": frame.index[-1],
                     **result.metrics})
        predictions.append(frame.assign(fold=number))
        if verbose:
            m = result.metrics
            print(f"fold {number}/{len(folds)}: accuracy {m['test_direction_accuracy']:.3f} "
                  f"(guess {m['baseline_majority_accuracy']:.3f}), "
                  f"rank corr {m['test_rank_corr']:+.3f}, best epoch {m['best_epoch']}")
    return WalkForwardResult(pd.DataFrame(rows).set_index("fold"), pd.concat(predictions))
