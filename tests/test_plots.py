"""Smoke tests: chart functions build a figure without raising."""
import matplotlib

matplotlib.use("Agg")  # draw off-screen, no window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fxml.evaluation import plots


def _data(n=300):
    rng = np.random.default_rng(1)
    features = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    target = pd.Series(rng.normal(size=n))
    return features, target


def test_feature_distributions_returns_a_figure():
    features, _ = _data()
    fig = plots.feature_distributions(features, ncols=3)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_feature_target_correlation_returns_a_figure():
    features, target = _data()
    fig = plots.feature_target_correlation(features, target, horizon=4)
    assert isinstance(fig, Figure)
    plt.close(fig)

def test_feature_correlation_heatmap_returns_a_figure():
    features, _ = _data()
    fig = plots.feature_correlation_heatmap(features)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_learning_curves_returns_a_figure():
    history = pd.DataFrame({
        "epoch": [1, 2, 3, 4],
        "train_loss": [1.0, 0.9, 0.85, 0.8], "val_loss": [1.0, 0.95, 0.97, 1.0],
        "train_direction_acc": [0.5, 0.52, 0.54, 0.56],
        "val_direction_acc": [0.5, 0.51, 0.5, 0.49],
    })
    fig = plots.learning_curves(history, best_epoch=2)
    assert isinstance(fig, Figure)
    plt.close(fig)

def test_fold_results_returns_a_figure():
    metrics = pd.DataFrame({
        "test_from": pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01"]),
        "test_direction_accuracy": [0.52, 0.49, 0.51],
        "baseline_majority_accuracy": [0.49, 0.51, 0.50],
        "test_rank_corr": [0.09, -0.01, 0.02],
        "n_test": [3100, 3100, 3100],
    })
    fig = plots.fold_results(metrics, horizon=4)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_equity_curve_returns_a_figure():
    index = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    gross = pd.Series([0.0, 0.001, 0.002, 0.001, 0.003], index=index)
    fig = plots.equity_curve(gross, gross - 0.001)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_threshold_sweep_chart_returns_a_figure():
    sweep = pd.DataFrame(
        {"gross_bps_per_trade": [0.1, 0.3, 0.8], "net_bps_per_trade": [-0.5, -0.3, 0.1],
         "n_trades": [4000, 2000, 800]},
        index=pd.Index([0.0, 0.01, 0.03], name="threshold"))
    fig = plots.threshold_sweep_chart(sweep)
    assert isinstance(fig, Figure)
    plt.close(fig)

def test_fold_lines_returns_a_figure():
    frame = pd.DataFrame({"persistence": [0.2, 0.3], "HAR": [0.4, 0.45], "HAR+": [0.41, 0.46]},
                         index=pd.Index([1, 2], name="fold"))
    fig = plots.fold_lines(frame, ["2020-01", "2020-07"], "title", "R2")
    assert isinstance(fig, Figure)
    plt.close(fig)
