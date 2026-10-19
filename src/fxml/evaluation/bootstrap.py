"""Block bootstrap confidence interval for an annualised Sharpe ratio."""
import numpy as np
import pandas as pd


def block_bootstrap_sharpe(returns: pd.Series, periods_per_year: float = 260, block: int = 20,
                           n_resamples: int = 2000, seed: int = 0, level: float = 0.90):
    """Interval for the Sharpe ratio by resampling runs of consecutive days.

    Blocks of `block` days wrap around the end of the series, so volatility clustering and
    short-term dependence survive the resampling. Returns (low, high) at the given level.
    """
    values = np.asarray(returns.dropna(), dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)  # round up
    starts = rng.integers(0, n, size=(n_resamples, n_blocks))
    index = ((starts[:, :, None] + np.arange(block)) % n).reshape(n_resamples, -1)[:, :n]
    samples = values[index]
    sharpes = samples.mean(axis=1) / samples.std(axis=1, ddof=1) * np.sqrt(periods_per_year)
    tail = (1 - level) / 2
    low, high = np.quantile(sharpes, [tail, 1 - tail])
    return float(low), float(high)
