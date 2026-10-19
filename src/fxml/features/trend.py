"""Trend-following signal for daily bars."""
import numpy as np
import pandas as pd

TREND_LOOKBACKS = (21, 63, 126)  # about 1, 3 and 6 trading months of daily bars


def trend_signal(close: pd.Series, lookbacks=TREND_LOOKBACKS) -> pd.Series:
    """Average of the signs of the trailing log return over each lookback.

    +1 when every speed is rising, -1 when every speed is falling, in between when they disagree.
    Uses closes up to and including each day only; NaN until the longest lookback has history.
    """
    log_close = np.log(close)
    signs = pd.concat([np.sign(log_close.diff(n)) for n in lookbacks], axis=1)
    return signs.mean(axis=1, skipna=False)
