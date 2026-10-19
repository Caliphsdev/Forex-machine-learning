"""Tests for evaluation metrics."""
import numpy as np
import pytest

from fxml.evaluation.metrics import oos_r2

from fxml.evaluation.metrics import (
    directional_accuracy, majority_baseline_accuracy, mae, rmse,
)


def test_directional_accuracy():
    assert directional_accuracy([1, 0, 1, 1], [0.9, 0.2, 0.4, 0.6]) == 0.75


def test_rmse_and_mae():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert mae([0, 0], [3, 4]) == pytest.approx(3.5)


def test_majority_baseline_uses_the_training_majority_on_the_test_set():
    assert majority_baseline_accuracy([1, 1, 0], [1, 0, 0, 0]) == 0.25


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        rmse([1, 2, 3], [1, 2])

def test_oos_r2_is_one_for_perfect_zero_for_benchmark_negative_for_worse():
    actual = [1.0, 2.0, 3.0]
    assert oos_r2(actual, [1.0, 2.0, 3.0], benchmark=2.0) == pytest.approx(1.0)
    assert oos_r2(actual, [2.0, 2.0, 2.0], benchmark=2.0) == pytest.approx(0.0)
    assert oos_r2(actual, [3.0, 2.0, 1.0], benchmark=2.0) == pytest.approx(-3.0)
