"""Combines chosen indicators into the final feature matrix."""
import numpy as np
import pandas as pd

from fxml.features import indicators as ind

FEATURE_COLUMNS = [
    "ret_1", "ret_4", "ret_24",
    "log_atr", "log_range", "log_vol_24", "vol_rel",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
CYCLICAL_COLUMNS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def _log(values: pd.Series) -> pd.Series:
    """Log of a positive ratio; floored at 1e-6 so a zero never becomes -inf."""
    return np.log(values.clip(lower=1e-6))


def _cyclical(values, period: int) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * np.asarray(values) / period
    return np.sin(angle), np.cos(angle)


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Bars (UTC DatetimeIndex, OHLC + tick_volume) -> feature table, same index.

    Warm-up rows are NaN. Every feature uses only the current and past bars.
    """
    close = bars["close"]
    ret_1 = ind.log_return(close, 1)

    features = pd.DataFrame(index=bars.index)
    features["ret_1"] = ret_1
    features["ret_4"] = ind.log_return(close, 4)
    features["ret_24"] = ind.log_return(close, 24)
    features["log_atr"] = _log(ind.atr(bars["high"], bars["low"], close, 14) / close)
    features["log_range"] = _log((bars["high"] - bars["low"]) / close)
    features["log_vol_24"] = _log(ret_1.rolling(24).std())

    features["vol_rel"] = np.log(
        bars["tick_volume"] / bars["tick_volume"].rolling(24).mean()
    )
    features["hour_sin"], features["hour_cos"] = _cyclical(bars.index.hour, 24)
    features["dow_sin"], features["dow_cos"] = _cyclical(bars.index.dayofweek, 7)

    return features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
