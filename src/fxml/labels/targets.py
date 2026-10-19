"""Defines what the model predicts (future return, direction)."""
import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Log return from this bar's close to the close `horizon` bars later.

    The last `horizon` rows are NaN because the future does not exist yet.
    """
    log_close = np.log(close)
    return log_close.shift(-horizon) - log_close


def direction(fwd_return: pd.Series) -> pd.Series:
    """1.0 if the forward return is positive, else 0.0. NaN stays NaN."""
    return (fwd_return > 0).astype(float).where(fwd_return.notna())


def forward_log_vol(close: pd.Series, horizon: int) -> pd.Series:
    """Log RMS of the next `horizon` 1-bar log returns (bars t+1 .. t+horizon).

    The last `horizon` rows are NaN because that future does not exist yet.
    """
    squared = np.log(close).diff() ** 2
    future_mean = squared.rolling(horizon).mean().shift(-horizon)
    return 0.5 * np.log(future_mean.clip(lower=1e-12))
