"""The LSTM network definition (nn.Module)."""
import torch
from torch import nn


class TwoHeadLSTM(nn.Module):
    """Window of features -> (predicted return, direction logit).

    The direction head returns a logit, not a probability: BCEWithLogitsLoss
    is numerically safer, and sigmoid(logit) gives the probability at inference.
    """

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # input is (batch, time, features)
            dropout=dropout if num_layers > 1 else 0.0,  # nn.LSTM only drops between layers
        )
        self.dropout = nn.Dropout(dropout)
        self.return_head = nn.Linear(hidden_size, 1)
        self.direction_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.lstm(x)          # (batch, time, hidden)
        last = self.dropout(output[:, -1, :])  # state after the final bar
        return self.return_head(last).squeeze(-1), self.direction_head(last).squeeze(-1)
