"""Tests for the block bootstrap confidence interval on a Sharpe ratio."""
import numpy as np
import pandas as pd

from fxml.evaluation.bootstrap import block_bootstrap_sharpe


def daily_returns(n: int, mean: float = 0.0, seed: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mean, 0.006, n))


def sample_sharpe(returns: pd.Series) -> float:
    return returns.mean() / returns.std() * np.sqrt(260)


def test_the_same_seed_gives_the_same_interval():
    returns = daily_returns(1000)
    assert block_bootstrap_sharpe(returns, seed=3) == block_bootstrap_sharpe(returns, seed=3)


def test_the_interval_brackets_the_sample_sharpe():
    returns = daily_returns(1000, mean=0.0003)
    low, high = block_bootstrap_sharpe(returns)
    assert low < sample_sharpe(returns) < high


def test_more_data_gives_a_narrower_interval():
    short_low, short_high = block_bootstrap_sharpe(daily_returns(250, mean=0.0003))
    long_low, long_high = block_bootstrap_sharpe(daily_returns(2500, mean=0.0003))
    assert (long_high - long_low) < (short_high - short_low)


def test_pure_noise_includes_zero_and_a_strong_edge_does_not():
    noise_low, noise_high = block_bootstrap_sharpe(daily_returns(2500, mean=0.0))
    edge_low, _ = block_bootstrap_sharpe(daily_returns(2500, mean=0.0006))
    assert noise_low < 0 < noise_high
    assert edge_low > 0


def test_missing_returns_are_ignored():
    returns = daily_returns(1000, mean=0.0003)
    padded = pd.concat([pd.Series([np.nan] * 5), returns], ignore_index=True)
    assert block_bootstrap_sharpe(padded, seed=1) == block_bootstrap_sharpe(returns, seed=1)
