"""Tests for the training loop (tiny CPU problems, fast)."""
import pandas as pd
import pytest
import torch

from fxml.datasets.windows import WindowDataset
from fxml.models.lstm import TwoHeadLSTM
from fxml.evaluation.metrics import directional_accuracy
from fxml.training.trainer import compute_losses, evaluate, predict, train_model

CPU = torch.device("cpu")
N_FEATURES, LOOKBACK = 3, 4


def make_dataset(n: int, seed: int, learnable: bool) -> WindowDataset:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, N_FEATURES, generator=gen)
    if learnable:
        y_return = x[:, 0].clone()            # the answer is in the last row's first feature
        y_direction = (x[:, 0] > 0).float()
    else:
        y_return = torch.randn(n, generator=gen)  # pure noise: nothing to learn
        y_direction = (torch.rand(n, generator=gen) > 0.5).float()
    return WindowDataset(x.numpy(), y_return.numpy(), y_direction.numpy(),
                         lookback=LOOKBACK, start=LOOKBACK - 1, end=n)


def fresh_model(seed: int = 0) -> TwoHeadLSTM:
    torch.manual_seed(seed)
    return TwoHeadLSTM(N_FEATURES, hidden_size=16)


def test_compute_losses_adds_weighted_direction_loss_to_return_loss():
    pred, y = torch.tensor([0.0]), torch.tensor([1.0])
    logit, y_dir = torch.tensor([0.0]), torch.tensor([1.0])
    total, ret, direction = compute_losses(pred, logit, y, y_dir, direction_weight=2.0)
    assert ret.item() == pytest.approx(1.0)
    assert direction.item() == pytest.approx(0.6931, abs=1e-4)  # ln 2
    assert total.item() == pytest.approx(1.0 + 2 * 0.6931, abs=1e-3)


def test_training_learns_a_problem_that_has_signal():
    train_ds, val_ds = make_dataset(400, 0, True), make_dataset(200, 1, True)
    result = train_model(fresh_model(), train_ds, val_ds, epochs=30, batch_size=64,
                         lr=1e-2, patience=30, device=CPU)
    history = result.history
    assert history["train_loss"].iloc[-1] < 0.5 * history["train_loss"].iloc[0]
    assert history["val_direction_acc"].max() > 0.85


def test_history_has_one_row_per_epoch_with_expected_columns():
    train_ds, val_ds = make_dataset(100, 0, True), make_dataset(60, 1, True)
    result = train_model(fresh_model(), train_ds, val_ds, epochs=3, patience=10, device=CPU)
    assert len(result.history) == 3
    assert result.history["epoch"].tolist() == [1, 2, 3]
    for column in ["train_loss", "val_loss", "train_return_loss", "val_return_loss",
                   "train_direction_loss", "val_direction_loss",
                   "train_direction_acc", "val_direction_acc"]:
        assert column in result.history.columns


def test_early_stopping_halts_training_on_pure_noise():
    train_ds, val_ds = make_dataset(300, 0, False), make_dataset(150, 1, False)
    result = train_model(fresh_model(), train_ds, val_ds, epochs=100, batch_size=64,
                         lr=1e-2, patience=3, device=CPU)
    assert len(result.history) < 100
    assert len(result.history) == result.best_epoch + 3  # stopped exactly `patience` later


def test_best_weights_are_restored_after_training():
    train_ds, val_ds = make_dataset(300, 0, False), make_dataset(150, 1, False)
    model = fresh_model()
    result = train_model(model, train_ds, val_ds, epochs=100, batch_size=64,
                         lr=1e-2, patience=3, device=CPU)
    best_val_loss = result.history["val_loss"].iloc[result.best_epoch - 1]
    assert evaluate(model, val_ds, device=CPU)["loss"] == pytest.approx(best_val_loss, abs=1e-5)


def test_same_seed_gives_identical_history():
    train_ds, val_ds = make_dataset(100, 0, True), make_dataset(60, 1, True)
    first = train_model(fresh_model(), train_ds, val_ds, epochs=3, device=CPU, seed=7)
    second = train_model(fresh_model(), train_ds, val_ds, epochs=3, device=CPU, seed=7)
    pd.testing.assert_frame_equal(first.history, second.history)


def test_predict_returns_one_value_per_sample_with_valid_probabilities():
    ds = make_dataset(50, 0, True)
    pred_return, prob_up = predict(fresh_model(), ds, device=CPU)
    assert pred_return.shape == (len(ds),)
    assert prob_up.shape == (len(ds),)
    assert ((prob_up >= 0) & (prob_up <= 1)).all()


def test_predict_agrees_with_evaluate_on_direction_accuracy():
    ds = make_dataset(50, 0, True)
    model = fresh_model()
    _, prob_up = predict(model, ds, device=CPU)
    actual = ds.y_direction[ds.start:ds.end].numpy()
    assert directional_accuracy(actual, prob_up) == pytest.approx(
        evaluate(model, ds, device=CPU)["direction_acc"])
