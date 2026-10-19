"""Pure functions: returns, ATR, volatility, etc. Series in, Series out."""
import numpy as np
import pandas as pd


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log return over `periods` bars: ln(close_t / close_{t-periods})."""
    return np.log(close).diff(periods)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Largest of: high-low, |high-prev_close|, |low-prev_close|."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average true range with Wilder smoothing. NaN until `period` bars exist."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def log_realised_vol(close: pd.Series, window: int) -> pd.Series:
    """Log of the root-mean-square 1-bar log return over the last `window` bars."""
    squared = log_return(close, 1) ** 2
    return 0.5 * np.log(squared.rolling(window).mean().clip(lower=1e-12))

