from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from musak_model.model.config import TransformerConfig


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, *, hidden_size: int, max_length: int) -> None:
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        dimensions = torch.arange(0, hidden_size, 2)
        angles = positions / (10000 ** (dimensions / hidden_size))
        encoding = torch.zeros(max_length, hidden_size)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles)
        self.register_buffer("_encoding", encoding.unsqueeze(0))

    def forward(self, sequence_length: int) -> Tensor:
        encoding: Tensor = self._encoding  # type: ignore[assignment]
        return encoding[:, :sequence_length]


class BeatStrengthBias(nn.Module):
    def __init__(self, *, hidden_size: int, max_length: int) -> None:
        super().__init__()
        self._embedding = nn.Embedding(max_length, hidden_size)

    def forward(self, beat_positions: Tensor) -> Tensor:
        return cast(Tensor, self._embedding(beat_positions))


class CausalTransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self._config = config

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self._decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self._position_embedding = SinusoidalPositionEmbedding(
            hidden_size=config.hidden_size,
            max_length=config.max_sequence_length,
        )

    def forward(self, target_embeddings: Tensor, bar_context: Tensor) -> Tensor:
        # target_embeddings: (batch, target_seq, hidden) — token sequence being generated
        # bar_context:       (batch, num_bars, hidden)   — bar-level latents from GRU
        sequence_length = target_embeddings.size(1)
        positional = self._position_embedding(sequence_length)
        target = target_embeddings + positional

        causal_mask = self._causal_mask(sequence_length, device=target.device)
        return cast(Tensor, self._decoder(target, bar_context, tgt_mask=causal_mask, tgt_is_causal=True))

    @staticmethod
    def _causal_mask(size: int, *, device: torch.device) -> Tensor:
        return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

    @property
    def hidden_size(self) -> int:
        return self._config.hidden_size
