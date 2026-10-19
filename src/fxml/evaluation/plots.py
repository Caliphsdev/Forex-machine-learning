"""Reusable chart functions. Each returns a matplotlib Figure; notebooks decide what to show."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from fxml import config
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def apply_style() -> None:
    """Quiet, readable defaults: thin lines, hairline grid, no top/right borders."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "lines.linewidth": 2,
    })


def feature_distributions(scaled: pd.DataFrame, clip: float = 10.0, ncols: int = 4) -> Figure:
    """One histogram per scaled feature, with the share of values beyond +-clip."""
    n = len(scaled.columns)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.3 * nrows), squeeze=False)
    for ax, column in zip(axes.flat, scaled.columns):
        values = scaled[column].dropna()
        outside = (values.abs() > clip).mean() * 100
        ax.hist(np.clip(values, -clip, clip), bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
        ax.set_title(column)
        ax.set_yticks([])
        ax.text(0.98, 0.92, f"{outside:.2f}% beyond +-{clip:g}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=MUTED)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle("Scaled feature distributions (clipped for display)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def feature_target_correlation(features: pd.DataFrame, target: pd.Series, horizon: int = 1) -> Figure:
    """Spearman correlation of each feature with the target, plus a rough noise band.

    Overlapping targets make neighbouring rows dependent, so the band is widened
    by sqrt(horizon). Bars inside the band are indistinguishable from zero.
    """
    data = features.join(target.rename("target")).dropna()
    corr = data.drop(columns="target").corrwith(data["target"], method="spearman").sort_values()
    band = 2 * np.sqrt(horizon / len(data))

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(corr) + 1.2))
    ax.axvspan(-band, band, color=GRID, alpha=0.8, zorder=0, label="noise band")
    ax.barh(corr.index, corr.values, color=BLUE, height=0.6, zorder=2)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_title("Which features relate to the future return? (Spearman)")
    ax.set_xlabel("correlation with forward return")
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig

DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])

def feature_correlation_heatmap(features: pd.DataFrame) -> Figure:
    """Feature-vs-feature correlation. Blue = negative, red = positive, grey = none."""
    corr = features.corr(method="spearman")
    size = 0.6 * len(corr) + 2
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(corr, ax=ax, cmap=DIVERGING, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8, "color": INK},
                linewidths=1, linecolor=SURFACE, square=True,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Do any features say the same thing? (Spearman)")
    ax.grid(False)
    fig.tight_layout()
    return fig

def learning_curves(history: pd.DataFrame, best_epoch: int) -> Figure:
    """Loss and direction accuracy per epoch, train vs validation.

    Training keeps improving by design. What matters is whether validation
    follows it, and where it stops improving (the dashed line is the epoch
    whose weights were kept).
    """
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    panels = [
        (ax_loss, "train_loss", "val_loss", "Loss (lower is better)"),
        (ax_acc, "train_direction_acc", "val_direction_acc",
         "Direction accuracy (0.5 = coin flip)"),
    ]
    for ax, train_col, val_col, title in panels:
        ax.plot(history["epoch"], history[train_col], color=BLUE, label="train")
        ax.plot(history["epoch"], history[val_col], color=ORANGE, label="validation")
        ax.axvline(best_epoch, color=AXIS, linewidth=1, linestyle="--")
        ax.set_title(title)
        ax.legend(frameon=False, labelcolor=INK_2)
    ax_acc.axhline(0.5, color=MUTED, linewidth=1)
    ax_acc.set_xlabel("epoch")
    fig.tight_layout()
    return fig

def fold_results(metrics: pd.DataFrame, horizon: int = config.LABEL_HORIZON) -> Figure:
    """Per-fold direction accuracy vs the majority guess, and rank correlation vs noise."""
    x = np.arange(len(metrics))
    labels = [t.strftime("%Y-%m") for t in metrics["test_from"]]
    fig, (ax_acc, ax_corr) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    ax_acc.plot(x, metrics["test_direction_accuracy"], color=BLUE, marker="o", label="model")
    ax_acc.plot(x, metrics["baseline_majority_accuracy"], color=ORANGE, marker="o",
                label="majority guess")
    ax_acc.axhline(0.5, color=MUTED, linewidth=1)
    mean_acc = metrics["test_direction_accuracy"].mean()
    ax_acc.set_title(f"Direction accuracy per fold (model mean {mean_acc:.3f})")
    ax_acc.legend(frameon=False, labelcolor=INK_2)

    band = 2 * np.sqrt(horizon / metrics["n_test"].mean())
    ax_corr.axhspan(-band, band, color=GRID, alpha=0.8, zorder=0, label="noise band")
    ax_corr.bar(x, metrics["test_rank_corr"], color=BLUE, width=0.6, zorder=2)
    ax_corr.axhline(0, color=AXIS, linewidth=1)
    mean_corr = metrics["test_rank_corr"].mean()
    ax_corr.set_title(f"Rank correlation of prediction vs outcome (mean {mean_corr:+.3f})")
    ax_corr.legend(frameon=False, labelcolor=INK_2, loc="lower right")
    ax_corr.set_xticks(x, labels, rotation=45, ha="right")
    ax_corr.set_xlabel("test window start")
    fig.tight_layout()
    return fig

def equity_curve(gross: pd.Series, net: pd.Series) -> Figure:
    """Cumulative return of trading the signals, before and after costs."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gross.index, gross * 100, color=BLUE, label="before costs")
    ax.plot(net.index, net * 100, color=ORANGE, label="after spread")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_title("Cumulative return of trading the signals")
    ax.set_ylabel("log return, %")
    ax.legend(frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    return fig


def threshold_sweep_chart(sweep: pd.DataFrame) -> Figure:
    """Return per trade (before and after costs) and trade count by confidence threshold."""
    fig, (ax_bps, ax_n) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_bps.plot(sweep.index, sweep["gross_bps_per_trade"], color=BLUE, marker="o",
                label="before costs")
    ax_bps.plot(sweep.index, sweep["net_bps_per_trade"], color=ORANGE, marker="o",
                label="after costs")
    ax_bps.axhline(0, color=AXIS, linewidth=1)
    ax_bps.set_title("Average return per trade (bps) by confidence threshold")
    ax_bps.legend(frameon=False, labelcolor=INK_2)
    ax_n.plot(sweep.index, sweep["n_trades"], color=BLUE, marker="o")
    ax_n.set_title("Number of trades")
    ax_n.set_xlabel("minimum |P(up) - 0.5| required to trade")
    fig.tight_layout()
    return fig


AQUA = "#1baf7a"  # third slot of the categorical palette


def fold_lines(frame: pd.DataFrame, labels: list[str], title: str, ylabel: str,
               zero_line: bool = True) -> Figure:
    """One line per column across walk-forward folds (x labels are the test start dates)."""
    colors = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(frame))
    for color, column in zip(colors, frame.columns):
        ax.plot(x, frame[column], color=color, marker="o", label=column)
    if zero_line:
        ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("test window start")
    ax.legend(frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    return fig
