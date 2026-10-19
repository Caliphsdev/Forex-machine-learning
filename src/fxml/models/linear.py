"""Ordinary least squares baseline: the simple model a neural net must beat."""
import numpy as np
import pandas as pd


class LinearModel:
    """OLS with an intercept. Columns are matched by name at predict time."""

    def __init__(self) -> None:
        self.columns_: list[str] | None = None
        self.coef_: np.ndarray | None = None  # coef_[0] is the intercept

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LinearModel":
        self.columns_ = list(features.columns)
        design = np.column_stack([np.ones(len(features)), features.to_numpy(dtype=float)])
        self.coef_, *_ = np.linalg.lstsq(design, target.to_numpy(dtype=float), rcond=None)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict()")
        design = np.column_stack(
            [np.ones(len(features)), features[self.columns_].to_numpy(dtype=float)]
        )
        return design @ self.coef_
