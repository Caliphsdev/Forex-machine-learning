"""Builds the modelling table: features + targets, aligned by time, NaN-free."""
import pandas as pd

from fxml.features.builder import FEATURE_COLUMNS, build_features
from fxml.features.volatility import VOL_FEATURE_COLUMNS, build_vol_features
from fxml.labels.targets import direction, forward_log_vol, forward_return

TARGET_COLUMNS = ["target_return", "target_direction"]


def build_table(bars: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Bars -> one row per bar with FEATURE_COLUMNS then TARGET_COLUMNS.

    Rows with any NaN are dropped: the warm-up rows at the start (indicators
    need history) and the last `horizon` rows (their future does not exist yet).
    """
    table = build_features(bars)
    fwd = forward_return(bars["close"], horizon)
    table["target_return"] = fwd
    table["target_direction"] = direction(fwd)
    return table[FEATURE_COLUMNS + TARGET_COLUMNS].dropna()


VOL_TARGET = "target_log_vol"


def build_vol_table(bars: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Bars -> VOL_FEATURE_COLUMNS plus the forward log-vol target, NaN rows dropped."""
    table = build_vol_features(bars)
    table[VOL_TARGET] = forward_log_vol(bars["close"], horizon)
    return table[VOL_FEATURE_COLUMNS + [VOL_TARGET]].dropna()
