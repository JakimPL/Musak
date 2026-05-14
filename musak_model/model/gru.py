from typing import cast

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

    def forward(
        self,
        bar_embeddings: Tensor,
        *,
        lengths: Tensor | None = None,
    ) -> Tensor:
        # bar_embeddings: (batch, tokens_in_bar, hidden)
        # lengths: (batch,) — actual token counts per bar; enables packing away padding
        # returns: (batch, hidden) — single latent vector per bar
        hidden = self._encode_with_optional_packing(bar_embeddings=bar_embeddings, lengths=lengths)

        # hidden: (num_layers * directions, batch, hidden) — take last layer
        last_hidden = hidden[-1]
        return cast(Tensor, self._projection(last_hidden))

    def _encode_with_optional_packing(
        self,
        *,
        bar_embeddings: Tensor,
        lengths: Tensor | None,
    ) -> Tensor:
        if lengths is not None:
            packed = pack_padded_sequence(bar_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self._gru(packed)
        else:
            _, hidden = self._gru(bar_embeddings)

        return cast(Tensor, hidden)


class BarPrefixGRUEncoder(nn.Module):
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

    def forward(
        self,
        bar_embeddings: Tensor,
        *,
        lengths: Tensor,
    ) -> Tensor:
        packed = pack_padded_sequence(bar_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self._gru(packed)
        padded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=bar_embeddings.size(1))
        return cast(Tensor, self._projection(padded))
