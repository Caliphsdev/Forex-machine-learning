"""Tests for the windowed PyTorch dataset."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fxml.datasets.windows import WindowDataset

N = 12


def make_arrays():
    rows = np.arange(N, dtype=np.float32)
    x = np.stack([rows, rows * 10], axis=1)  # column 0 = row number, column 1 = 10x
    return x, rows / 100, rows % 2


def make_ds(start=5, end=8, lookback=3):
    x, y_return, y_direction = make_arrays()
    return WindowDataset(x, y_return, y_direction, lookback=lookback, start=start, end=end)


def test_window_holds_lookback_rows_ending_at_the_anchor():
    ds = make_ds()
    assert ds[0][0][:, 0].tolist() == [3.0, 4.0, 5.0]
    assert ds[0][0][:, 1].tolist() == [30.0, 40.0, 50.0]
    assert ds[2][0][:, 0].tolist() == [5.0, 6.0, 7.0]


def test_targets_belong_to_the_anchor_row():
    _, y_return, y_direction = make_ds()[1]  # anchor row 6
    assert y_return.item() == pytest.approx(0.06)
    assert y_direction.item() == 0.0


def test_length_shape_and_dtype():
    ds = make_ds()
    window, y_return, y_direction = ds[0]
    assert len(ds) == 3
    assert window.shape == (3, 2)
    assert window.dtype == torch.float32
    assert y_return.dtype == torch.float32


def test_window_never_contains_rows_after_the_anchor():
    ds = make_ds(start=2, end=12, lookback=3)
    for k in range(len(ds)):
        window, _, _ = ds[k]
        assert window[:, 0].max().item() == ds.start + k


def test_start_without_enough_history_raises():
    with pytest.raises(ValueError):
        make_ds(start=1, end=8, lookback=3)


def test_bad_ranges_raise():
    with pytest.raises(ValueError):
        make_ds(start=5, end=13)  # beyond the data
    with pytest.raises(ValueError):
        make_ds(start=8, end=8)   # empty


def test_nan_inputs_raise():
    x, y_return, y_direction = make_arrays()
    x[4, 0] = np.nan
    with pytest.raises(ValueError):
        WindowDataset(x, y_return, y_direction, lookback=3, start=5, end=8)


def test_dataloader_batches_have_expected_shape():
    ds = make_ds(start=2, end=12, lookback=3)
    x_batch, y_return_batch, _ = next(iter(DataLoader(ds, batch_size=4)))
    assert x_batch.shape == (4, 3, 2)
    assert y_return_batch.shape == (4,)
