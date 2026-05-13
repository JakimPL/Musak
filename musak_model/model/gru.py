from typing import cast

import torch.nn as nn
from torch import Tensor

from musak_model.model.config import GRUConfig


class BarGRUEncoder(nn.Module):
    def __init__(self, config: GRUConfig) -> None:
        super().__init__()
        self._gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        directions = 2 if config.bidirectional else 1
        out_size = config.hidden_size * directions
        self._projection = nn.Linear(out_size, config.hidden_size)

    def forward(self, bar_embeddings: Tensor) -> Tensor:
        # bar_embeddings: (batch, tokens_in_bar, hidden)
        # returns: (batch, hidden) — single latent vector per bar
        _, hidden = self._gru(bar_embeddings)
        # hidden: (num_layers * directions, batch, hidden) — take last layer
        last_hidden = hidden[-1]
        return cast(Tensor, self._projection(last_hidden))
