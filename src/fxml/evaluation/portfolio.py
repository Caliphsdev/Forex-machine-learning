"""Combine several strategies' per-bar returns into one equal-capital portfolio."""
import numpy as np
import pandas as pd


def combine_equal_weight(per_bar: dict[str, pd.Series]) -> pd.Series:
    """Average the strategies' per-bar returns on the timestamps that all of them share."""
    return pd.concat(per_bar, axis=1, join="inner").mean(axis=1)


def return_correlations(per_bar: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation of the strategies' per-bar returns, on their shared timestamps."""
    return pd.concat(per_bar, axis=1, join="inner").corr()


def portfolio_stats(returns: pd.Series, periods_per_year: float = 6200) -> dict:
    """Annualised Sharpe, total return and max drawdown of a per-bar return series."""
    std = returns.std()
    equity = returns.cumsum()
    return {
        "n_bars": len(returns),
        "sharpe": float(returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "total_return": float(equity.iloc[-1]),
        "max_drawdown": float((equity - equity.cummax()).min()),
    }
