from __future__ import annotations

from typing import Final, cast

import torch
import torch.nn as nn
from torch import Tensor

from musak_model.model.config import ModelConfig, TokenInputEmbeddingMode
from musak_model.model.output import FlatTokenAttributeBuffers
from musak_model.tokens.factorized import (
    ABSENT_ATTRIBUTE_ID,
    ACCIDENTAL_ATTRIBUTE_COUNT,
    DEGREE_ATTRIBUTE_COUNT,
    HAND_ATTRIBUTE_COUNT,
    OCTAVE_OFFSET_ATTRIBUTE_COUNT,
    TOKEN_KIND_COUNT,
    flat_vocabulary_attributes,
)

_ABSENT_EMBEDDING_ID: Final[int] = 0
_ACTIVE_ATTRIBUTE_OFFSET: Final[int] = 1


class TokenInputEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._mode = config.input.embedding_mode
        self._flat_embedding = nn.Embedding(config.vocabulary_size, config.transformer.hidden_size)
        match self._mode:
            case TokenInputEmbeddingMode.FLAT:
                return
            case TokenInputEmbeddingMode.FLAT_PLUS_FACTORIZED:
                self._register_flat_attribute_buffers(config)
                self._kind_embedding = nn.Embedding(TOKEN_KIND_COUNT, config.transformer.hidden_size)
                self._degree_embedding = _optional_attribute_embedding(
                    DEGREE_ATTRIBUTE_COUNT,
                    config.transformer.hidden_size,
                )
                self._accidental_embedding = _optional_attribute_embedding(
                    ACCIDENTAL_ATTRIBUTE_COUNT,
                    config.transformer.hidden_size,
                )
                self._octave_offset_embedding = _optional_attribute_embedding(
                    OCTAVE_OFFSET_ATTRIBUTE_COUNT,
                    config.transformer.hidden_size,
                )
                self._duration_embedding = _optional_attribute_embedding(
                    config.duration_vocabulary_size,
                    config.transformer.hidden_size,
                )
                self._hand_embedding = _optional_attribute_embedding(
                    HAND_ATTRIBUTE_COUNT,
                    config.transformer.hidden_size,
                )

    def forward(self, token_ids: Tensor) -> Tensor:
        embeddings = _embedding_output(self._flat_embedding, token_ids)
        match self._mode:
            case TokenInputEmbeddingMode.FLAT:
                return embeddings
            case TokenInputEmbeddingMode.FLAT_PLUS_FACTORIZED:
                attributes = self._flat_attribute_buffers(token_ids)
                return (
                    embeddings
                    + _embedding_output(
                        self._kind_embedding,
                        attributes.kind_ids,
                    )
                    + _embedding_output(
                        self._degree_embedding,
                        _optional_embedding_ids(attributes.degree_ids),
                    )
                    + _embedding_output(
                        self._accidental_embedding,
                        _optional_embedding_ids(attributes.accidental_ids),
                    )
                    + _embedding_output(
                        self._octave_offset_embedding,
                        _optional_embedding_ids(attributes.octave_offset_ids),
                    )
                    + _embedding_output(
                        self._duration_embedding,
                        _optional_embedding_ids(attributes.duration_ids),
                    )
                    + _embedding_output(self._hand_embedding, _optional_embedding_ids(attributes.hand_ids))
                )

    def _register_flat_attribute_buffers(self, config: ModelConfig) -> None:
        attributes = FlatTokenAttributeBuffers.from_attributes(
            flat_vocabulary_attributes(duration_vocabulary_size=config.duration_vocabulary_size)
        )
        expected_vocabulary_size = attributes.kind_ids.numel()
        if expected_vocabulary_size != config.vocabulary_size:
            raise ValueError(
                f"factorized input attribute table size {expected_vocabulary_size} does not match "
                f"vocabulary_size={config.vocabulary_size}"
            )

        self.register_buffer("_input_flat_kind_ids", attributes.kind_ids, persistent=False)
        self.register_buffer("_input_flat_degree_ids", attributes.degree_ids, persistent=False)
        self.register_buffer("_input_flat_accidental_ids", attributes.accidental_ids, persistent=False)
        self.register_buffer("_input_flat_octave_offset_ids", attributes.octave_offset_ids, persistent=False)
        self.register_buffer("_input_flat_duration_ids", attributes.duration_ids, persistent=False)
        self.register_buffer("_input_flat_hand_ids", attributes.hand_ids, persistent=False)

    def _flat_attribute_buffers(self, token_ids: Tensor) -> FlatTokenAttributeBuffers:
        return FlatTokenAttributeBuffers(
            kind_ids=cast(Tensor, self._input_flat_kind_ids)[token_ids],
            degree_ids=cast(Tensor, self._input_flat_degree_ids)[token_ids],
            accidental_ids=cast(Tensor, self._input_flat_accidental_ids)[token_ids],
            octave_offset_ids=cast(Tensor, self._input_flat_octave_offset_ids)[token_ids],
            duration_ids=cast(Tensor, self._input_flat_duration_ids)[token_ids],
            hand_ids=cast(Tensor, self._input_flat_hand_ids)[token_ids],
        )


def _optional_attribute_embedding(attribute_count: int, hidden_size: int) -> nn.Embedding:
    return nn.Embedding(
        attribute_count + _ACTIVE_ATTRIBUTE_OFFSET,
        hidden_size,
        padding_idx=_ABSENT_EMBEDDING_ID,
    )


def _embedding_output(embedding: nn.Embedding, token_ids: Tensor) -> Tensor:
    return cast(Tensor, embedding(token_ids))


def _optional_embedding_ids(attribute_ids: Tensor) -> Tensor:
    return torch.where(
        attribute_ids == ABSENT_ATTRIBUTE_ID,
        torch.zeros_like(attribute_ids),
        attribute_ids + _ACTIVE_ATTRIBUTE_OFFSET,
    )
