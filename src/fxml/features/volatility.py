"""Feature set for forecasting volatility (separate so direction experiments stay reproducible)."""
import pandas as pd

from fxml.features import indicators as ind
from fxml.features.builder import CYCLICAL_COLUMNS, build_features

RV_WINDOWS = (6, 24, 120, 500)  # hour-ish, day, week, month of H1 bars
SHARED_COLUMNS = ["log_range", "log_atr", "vol_rel", *CYCLICAL_COLUMNS]
VOL_FEATURE_COLUMNS = [f"log_rv_{w}" for w in RV_WINDOWS] + SHARED_COLUMNS


def build_vol_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Feature table for forecasting volatility. Warm-up rows are NaN; past data only."""
    close = bars["close"]
    shared = build_features(bars)
    features = pd.DataFrame(index=bars.index)
    for window in RV_WINDOWS:
        features[f"log_rv_{window}"] = ind.log_realised_vol(close, window)
    for column in SHARED_COLUMNS:
        features[column] = shared[column]
    return features[VOL_FEATURE_COLUMNS]
