"""Fit scaler on train only, apply to val/test, save/load with the model."""
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


class FeatureScaler:
    """Robust scaling: (x - median) / IQR, with parameters learned from train rows only."""

    def __init__(self, passthrough: Iterable[str] = ()) -> None:
        self.passthrough = tuple(passthrough)  # columns left unscaled (e.g. sin/cos)
        self.center_: pd.Series | None = None
        self.scale_: pd.Series | None = None

    def fit(self, features: pd.DataFrame) -> "FeatureScaler":
        self.center_ = features.median()
        iqr = features.quantile(0.75) - features.quantile(0.25)
        self.scale_ = iqr.where(iqr > 0, 1.0)  # constant column: avoid divide-by-zero
        keep = [c for c in self.passthrough if c in features.columns]
        self.center_.loc[keep] = 0.0
        self.scale_.loc[keep] = 1.0
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before transform()")
        missing = [c for c in self.center_.index if c not in features.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return (features[self.center_.index] - self.center_) / self.scale_

    def save(self, path: Path | str) -> None:
        payload = {
            "columns": list(self.center_.index),
            "center": self.center_.tolist(),
            "scale": self.scale_.tolist(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "FeatureScaler":
        payload = json.loads(Path(path).read_text())
        scaler = cls()
        scaler.center_ = pd.Series(payload["center"], index=payload["columns"])
        scaler.scale_ = pd.Series(payload["scale"], index=payload["columns"])
        return scaler
