"""Tests for the two-headed LSTM and the model registry."""
import pytest
import torch

from fxml.models.lstm import TwoHeadLSTM
from fxml.models.registry import build_model

N_FEATURES, LOOKBACK = 11, 48


def batch(size=5):
    return torch.randn(size, LOOKBACK, N_FEATURES)


def test_forward_gives_one_value_per_sample_from_each_head():
    model = TwoHeadLSTM(N_FEATURES)
    pred_return, direction_logit = model(batch(5))
    assert pred_return.shape == (5,)
    assert direction_logit.shape == (5,)


def test_samples_in_a_batch_do_not_affect_each_other():
    model = TwoHeadLSTM(N_FEATURES).eval()
    x = batch(5)
    full_return, full_logit = model(x)
    single_return, single_logit = model(x[:1])
    assert torch.allclose(full_return[0], single_return[0], atol=1e-6)
    assert torch.allclose(full_logit[0], single_logit[0], atol=1e-6)


def test_dropout_is_off_in_eval_and_on_in_train():
    model = TwoHeadLSTM(N_FEATURES, dropout=0.5)
    x = batch(8)
    model.eval()
    assert torch.equal(model(x)[0], model(x)[0])
    model.train()
    assert not torch.equal(model(x)[0], model(x)[0])


def test_gradients_reach_every_parameter():
    model = TwoHeadLSTM(N_FEATURES)
    pred_return, direction_logit = model(batch())
    (pred_return.sum() + direction_logit.sum()).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_default_model_is_small():
    # LSTM: 4*(11*64 + 64*64) weights + 2*4*64 biases = 19,712; two heads: 65 each
    model = TwoHeadLSTM(N_FEATURES, hidden_size=64)
    assert sum(p.numel() for p in model.parameters()) == 19_842


def test_registry_builds_a_model_by_name():
    model = build_model("lstm", n_features=N_FEATURES)
    assert isinstance(model, TwoHeadLSTM)


def test_registry_rejects_unknown_names():
    with pytest.raises(ValueError):
        build_model("transformer", n_features=N_FEATURES)
