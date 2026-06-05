from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from torch import Tensor, nn

from musak_model.rhythm_refiner.config import RhythmRefinerModelConfig
from musak_model.rhythm_refiner.dataset import RhythmRefinerBatch
from musak_model.rhythm_refiner.vocabulary import (
    COACTIVITY_TARGET_STATE_COUNT,
    RHYTHM_INPUT_STATE_COUNT,
    RHYTHM_TARGET_STATE_COUNT,
)


@dataclass(frozen=True)
class RhythmRefinerLogits:
    activity: Tensor
    coactivity: Tensor


class RhythmRefinerModel(nn.Module):
    def __init__(self, config: RhythmRefinerModelConfig) -> None:
        super().__init__()
        self._config = config
        hidden_size = config.hidden_size
        self._right_state_embedding = nn.Embedding(RHYTHM_INPUT_STATE_COUNT, hidden_size)
        self._left_state_embedding = nn.Embedding(RHYTHM_INPUT_STATE_COUNT, hidden_size)
        self._cell_index_embedding = nn.Embedding(config.max_cells_per_bar + 1, hidden_size)
        self._bar_index_embedding = nn.Embedding(config.max_bar_count + 1, hidden_size)
        self._distance_to_end_embedding = nn.Embedding(config.max_distance_cells + 1, hidden_size)
        self._bar_duration_cell_count_embedding = nn.Embedding(config.max_cells_per_bar + 1, hidden_size)
        self._time_numerator_embedding = nn.Embedding(config.max_time_numerator + 1, hidden_size)
        self._time_denominator_embedding = nn.Embedding(config.max_time_denominator + 1, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        self._activity_head = nn.Linear(hidden_size, 2 * RHYTHM_TARGET_STATE_COUNT)
        self._coactivity_head = nn.Linear(hidden_size, COACTIVITY_TARGET_STATE_COUNT)

    def forward(self, batch: RhythmRefinerBatch) -> RhythmRefinerLogits:
        hidden = self._input_embeddings(batch)
        encoded = self._encoder(hidden, src_key_padding_mask=batch.padding_mask)
        activity_logits = self._activity_head(encoded).view(
            encoded.size(0),
            encoded.size(1),
            2,
            RHYTHM_TARGET_STATE_COUNT,
        )
        return RhythmRefinerLogits(
            activity=activity_logits,
            coactivity=self._coactivity_head(encoded),
        )

    def _input_embeddings(self, batch: RhythmRefinerBatch) -> Tensor:
        right_state_ids = batch.input_state_ids[:, :, 0]
        left_state_ids = batch.input_state_ids[:, :, 1]
        time_numerator = self._time_numerator_embedding(batch.time_numerator_ids).unsqueeze(1)
        time_denominator = self._time_denominator_embedding(batch.time_denominator_ids).unsqueeze(1)
        hidden = (
            self._right_state_embedding(right_state_ids)
            + self._left_state_embedding(left_state_ids)
            + self._cell_index_embedding(batch.cell_index_ids)
            + self._bar_index_embedding(batch.bar_index_ids)
            + self._distance_to_end_embedding(batch.distance_to_end_ids)
            + self._bar_duration_cell_count_embedding(batch.bar_duration_cell_count_ids)
            + time_numerator
            + time_denominator
        )
        return cast(Tensor, hidden)
