"""PyTorch Dataset that turns a feature table into (lookback x features) windows."""
import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Serves (window, return_target, direction_target) for anchor rows [start, end).

    The window for anchor row i is rows i-lookback+1 .. i, so it ends on the
    anchor and never reaches into the future. Windows are views into one tensor.
    """

    def __init__(self, features, target_return, target_direction,
                 lookback: int, start: int, end: int) -> None:
        x = np.array(features, dtype=np.float32)
        y_return = np.array(target_return, dtype=np.float32)
        y_direction = np.array(target_direction, dtype=np.float32)
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if start < lookback - 1:
            raise ValueError(f"start={start} leaves fewer than {lookback} rows of history")
        if end > len(x) or start >= end:
            raise ValueError(f"invalid anchor range [{start}, {end}) for {len(x)} rows")
        if np.isnan(x).any() or np.isnan(y_return).any() or np.isnan(y_direction).any():
            raise ValueError("inputs contain NaN; drop incomplete rows first")

        self.x = torch.from_numpy(x)
        self.y_return = torch.from_numpy(y_return)
        self.y_direction = torch.from_numpy(y_direction)
        self.lookback, self.start, self.end = lookback, start, end

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, k: int):
        if not 0 <= k < len(self):
            raise IndexError(k)
        i = self.start + k
        return self.x[i - self.lookback + 1 : i + 1], self.y_return[i], self.y_direction[i]
