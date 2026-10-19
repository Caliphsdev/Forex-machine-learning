"""RMSE, MAE, directional accuracy, and the baselines to compare them against."""
import numpy as np


def _as_pair(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    return actual, predicted


def directional_accuracy(actual_direction, prob_up) -> float:
    """Share of samples where (prob_up > 0.5) matches the actual up/down label."""
    actual, prob = _as_pair(actual_direction, prob_up)
    return float(np.mean((prob > 0.5) == (actual > 0.5)))


def rmse(actual, predicted) -> float:
    actual, predicted = _as_pair(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted) -> float:
    actual, predicted = _as_pair(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))


def majority_baseline_accuracy(train_direction, test_direction) -> float:
    """Accuracy of always guessing whichever direction was more common in training."""
    guess_up = np.mean(np.asarray(train_direction, dtype=float)) >= 0.5
    test = np.asarray(test_direction, dtype=float) > 0.5
    return float(np.mean(test == guess_up))

def oos_r2(actual, predicted, benchmark) -> float:
    """1 - SSE(model) / SSE(benchmark). 0 = no better than the benchmark prediction
    (normally the training mean), 1 = perfect, negative = worse than the benchmark."""
    actual, predicted = _as_pair(actual, predicted)
    sse_model = np.sum((actual - predicted) ** 2)
    sse_benchmark = np.sum((actual - benchmark) ** 2)
    return float(1 - sse_model / sse_benchmark)
