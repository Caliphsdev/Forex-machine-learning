"""Tests that splits never leak future data."""
import pytest

from fxml.datasets.splits import Fold, split_validation, walk_forward_splits

def test_known_example():
    folds = walk_forward_splits(n_samples=100, train_size=40, test_size=10, gap=2)
    assert len(folds) == 5
    assert folds[0] == Fold(0, 40, 42, 52)
    assert folds[-1].test_end == 92


def test_test_windows_tile_without_overlap():
    folds = walk_forward_splits(100, 40, 10, 2)
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end == later.test_start


def test_train_ends_exactly_gap_before_test_starts():
    for fold in walk_forward_splits(100, 40, 10, 2):
        assert fold.test_start - fold.train_end == 2
        assert fold.test_end <= 100


def test_expanding_keeps_start_and_rolling_keeps_length():
    expanding = walk_forward_splits(100, 40, 10, 2, expanding=True)
    assert all(f.train_start == 0 for f in expanding)
    rolling = walk_forward_splits(100, 40, 10, 2, expanding=False)
    assert all(f.train_end - f.train_start == 40 for f in rolling)


def test_gap_equal_to_horizon_keeps_label_prices_out_of_test():
    horizon = 4
    for fold in walk_forward_splits(200, 80, 20, gap=horizon):
        last_price_used_by_train_labels = (fold.train_end - 1) + horizon
        assert last_price_used_by_train_labels < fold.test_start


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        walk_forward_splits(100, 0, 10, 2)
    with pytest.raises(ValueError):
        walk_forward_splits(100, 40, 10, -1)


def test_raises_when_there_is_not_enough_data():
    with pytest.raises(ValueError):
        walk_forward_splits(30, 40, 10, 2)

def test_split_validation_layout():
    assert split_validation(0, 100, val_size=20, gap=4) == ((0, 76), (80, 100))


def test_split_validation_rejects_impossible_sizes():
    with pytest.raises(ValueError):
        split_validation(0, 20, val_size=18, gap=4)
    with pytest.raises(ValueError):
        split_validation(0, 100, val_size=0, gap=4)
