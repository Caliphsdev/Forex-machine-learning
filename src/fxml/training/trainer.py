"""Training loop: epochs, GPU device, early stopping, checkpointing."""
import copy
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainResult:
    history: pd.DataFrame  # one row per epoch
    best_epoch: int        # 1-based epoch whose weights were restored


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_losses(pred_return, direction_logit, y_return, y_direction,
                   direction_weight: float = 1.0):
    """Total loss = return MSE + direction_weight * direction BCE (returns all three)."""
    return_loss = F.mse_loss(pred_return, y_return)
    direction_loss = F.binary_cross_entropy_with_logits(direction_logit, y_direction)
    return return_loss + direction_weight * direction_loss, return_loss, direction_loss


def _run_epoch(model, loader, device, direction_weight, optimizer=None, clip=1.0) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "return_loss": 0.0, "direction_loss": 0.0, "direction_acc": 0.0}
    count = 0
    with torch.set_grad_enabled(training):
        for x, y_return, y_direction in loader:
            x, y_return, y_direction = x.to(device), y_return.to(device), y_direction.to(device)
            pred_return, logit = model(x)
            loss, return_loss, direction_loss = compute_losses(
                pred_return, logit, y_return, y_direction, direction_weight
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            size = len(x)
            count += size
            totals["loss"] += loss.item() * size
            totals["return_loss"] += return_loss.item() * size
            totals["direction_loss"] += direction_loss.item() * size
            totals["direction_acc"] += ((logit > 0).float() == y_direction).sum().item()
    return {name: value / count for name, value in totals.items()}


def evaluate(model: nn.Module, dataset: Dataset, *, batch_size: int = 256,
             device: torch.device | None = None, direction_weight: float = 1.0) -> dict:
    """Mean losses and direction accuracy on a dataset, with dropout off."""
    device = device or pick_device()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return _run_epoch(model, loader, device, direction_weight)



def predict(model: nn.Module, dataset: Dataset, *, batch_size: int = 256,
            device: torch.device | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(predicted return, probability of up) for every sample, in dataset order."""
    device = device or pick_device()
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    returns, probabilities = [], []
    with torch.no_grad():
        for x, _, _ in loader:
            pred_return, logit = model(x.to(device))
            returns.append(pred_return.cpu())
            probabilities.append(torch.sigmoid(logit).cpu())
    return torch.cat(returns).numpy(), torch.cat(probabilities).numpy()


def train_model(model: nn.Module, train_ds: Dataset, val_ds: Dataset, *,
                epochs: int = 50, batch_size: int = 256, lr: float = 1e-3,
                patience: int = 8, direction_weight: float = 1.0,
                device: torch.device | None = None, seed: int = 0) -> TrainResult:
    """Train with early stopping on validation loss; restore the best weights."""
    device = device or pick_device()
    torch.manual_seed(seed)
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rows = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        train = _run_epoch(model, train_loader, device, direction_weight, optimizer)
        val = _run_epoch(model, val_loader, device, direction_weight)
        rows.append({"epoch": epoch,
                     **{f"train_{k}": v for k, v in train.items()},
                     **{f"val_{k}": v for k, v in val.items()}})
        if val["loss"] < best_loss:
            best_loss, best_epoch = val["loss"], epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return TrainResult(pd.DataFrame(rows), best_epoch)
