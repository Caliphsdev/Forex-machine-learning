"""Look up a model class by name so experiments can swap architectures from config."""
from torch import nn

from fxml.models.lstm import TwoHeadLSTM

MODELS: dict[str, type[nn.Module]] = {"lstm": TwoHeadLSTM}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)
