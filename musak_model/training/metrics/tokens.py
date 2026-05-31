from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from musak_model.tokens.factorized import ABSENT_ATTRIBUTE_ID, TokenKindId
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    StartToken,
)
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.dataset.factorized import (
    TokenAttributeTargetTensors,
    token_attribute_targets_from_token_ids,
)
from musak_model.training.metrics.schema import BatchMetrics


@dataclass(frozen=True)
class AttributeMatchCounts:
    duration_match_count: int | None = None
    duration_target_count: int | None = None
    degree_match_count: int | None = None
    degree_target_count: int | None = None
    accidental_match_count: int | None = None
    accidental_target_count: int | None = None
    octave_offset_match_count: int | None = None
    octave_offset_target_count: int | None = None
    hand_match_count: int | None = None
    hand_target_count: int | None = None


def batch_metrics_from_logits(
    logits: Tensor,
    *,
    target_token_ids: Tensor,
    token_padding_mask: Tensor,
    loss: Tensor,
    token_kind_ids: Tensor | None = None,
    token_attribute_lookup: TokenAttributeTargetTensors | None = None,
) -> BatchMetrics:
    valid_mask = ~token_padding_mask.reshape(-1)
    token_count = int(valid_mask.sum().item())
    if token_count == 0:
        raise ValueError("batch has no valid target tokens")

    predicted_token_ids = logits.argmax(dim=-1)
    exact_matches = (predicted_token_ids == target_token_ids) & ~token_padding_mask
    token_kind_match_count = _token_kind_match_count(
        predicted_token_ids,
        target_token_ids=target_token_ids,
        token_padding_mask=token_padding_mask,
        token_kind_ids=token_kind_ids,
    )
    attribute_counts = _attribute_match_counts(
        predicted_token_ids,
        target_token_ids=target_token_ids,
        token_padding_mask=token_padding_mask,
        token_attribute_lookup=token_attribute_lookup,
    )
    return BatchMetrics(
        loss=float(loss.detach().item()),
        token_count=token_count,
        exact_match_count=int(exact_matches.sum().item()),
        token_kind_match_count=token_kind_match_count,
        duration_match_count=attribute_counts.duration_match_count,
        duration_target_count=attribute_counts.duration_target_count,
        degree_match_count=attribute_counts.degree_match_count,
        degree_target_count=attribute_counts.degree_target_count,
        accidental_match_count=attribute_counts.accidental_match_count,
        accidental_target_count=attribute_counts.accidental_target_count,
        octave_offset_match_count=attribute_counts.octave_offset_match_count,
        octave_offset_target_count=attribute_counts.octave_offset_target_count,
        hand_match_count=attribute_counts.hand_match_count,
        hand_target_count=attribute_counts.hand_target_count,
    )


def build_token_kind_ids(vocabulary: TokenVocabulary) -> Tensor:
    return torch.tensor(
        [_token_kind_id(vocabulary.id_to_token(token_id)) for token_id in range(vocabulary.vocabulary_size)],
        dtype=torch.long,
    )


def build_token_attribute_lookup(vocabulary: TokenVocabulary) -> TokenAttributeTargetTensors:
    return token_attribute_targets_from_token_ids(
        torch.arange(vocabulary.vocabulary_size, dtype=torch.long),
        vocabulary=vocabulary,
    )


def _token_kind_match_count(
    predicted_token_ids: Tensor,
    *,
    target_token_ids: Tensor,
    token_padding_mask: Tensor,
    token_kind_ids: Tensor | None,
) -> int | None:
    if token_kind_ids is None:
        return None

    device_token_kind_ids = token_kind_ids.to(predicted_token_ids.device)
    predicted_kind_ids = device_token_kind_ids[predicted_token_ids]
    target_kind_ids = device_token_kind_ids[target_token_ids]
    kind_matches = (predicted_kind_ids == target_kind_ids) & ~token_padding_mask
    return int(kind_matches.sum().item())


def _attribute_match_counts(
    predicted_token_ids: Tensor,
    *,
    target_token_ids: Tensor,
    token_padding_mask: Tensor,
    token_attribute_lookup: TokenAttributeTargetTensors | None,
) -> AttributeMatchCounts:
    if token_attribute_lookup is None:
        return AttributeMatchCounts()

    predicted_attributes = _gather_token_attributes(
        predicted_token_ids,
        token_attribute_lookup=token_attribute_lookup,
    )
    target_attributes = _gather_token_attributes(
        target_token_ids,
        token_attribute_lookup=token_attribute_lookup,
    )
    duration_match_count, duration_target_count = _attribute_match_count(
        predicted_attributes.duration_ids,
        target_attributes.duration_ids,
        token_padding_mask=token_padding_mask,
    )
    degree_match_count, degree_target_count = _attribute_match_count(
        predicted_attributes.degree_ids,
        target_attributes.degree_ids,
        token_padding_mask=token_padding_mask,
    )
    accidental_match_count, accidental_target_count = _attribute_match_count(
        predicted_attributes.accidental_ids,
        target_attributes.accidental_ids,
        token_padding_mask=token_padding_mask,
    )
    octave_offset_match_count, octave_offset_target_count = _attribute_match_count(
        predicted_attributes.octave_offset_ids,
        target_attributes.octave_offset_ids,
        token_padding_mask=token_padding_mask,
    )
    hand_match_count, hand_target_count = _attribute_match_count(
        predicted_attributes.hand_ids,
        target_attributes.hand_ids,
        token_padding_mask=token_padding_mask,
    )
    return AttributeMatchCounts(
        duration_match_count=duration_match_count,
        duration_target_count=duration_target_count,
        degree_match_count=degree_match_count,
        degree_target_count=degree_target_count,
        accidental_match_count=accidental_match_count,
        accidental_target_count=accidental_target_count,
        octave_offset_match_count=octave_offset_match_count,
        octave_offset_target_count=octave_offset_target_count,
        hand_match_count=hand_match_count,
        hand_target_count=hand_target_count,
    )


def _gather_token_attributes(
    token_ids: Tensor,
    *,
    token_attribute_lookup: TokenAttributeTargetTensors,
) -> TokenAttributeTargetTensors:
    lookup = token_attribute_lookup.to(token_ids.device)
    return TokenAttributeTargetTensors(
        kind_ids=lookup.kind_ids[token_ids],
        degree_ids=lookup.degree_ids[token_ids],
        accidental_ids=lookup.accidental_ids[token_ids],
        octave_offset_ids=lookup.octave_offset_ids[token_ids],
        duration_ids=lookup.duration_ids[token_ids],
        hand_ids=lookup.hand_ids[token_ids],
    )


def _attribute_match_count(
    predicted_attribute_ids: Tensor,
    target_attribute_ids: Tensor,
    *,
    token_padding_mask: Tensor,
) -> tuple[int, int]:
    active_mask = (target_attribute_ids != ABSENT_ATTRIBUTE_ID) & ~token_padding_mask
    matches = (predicted_attribute_ids == target_attribute_ids) & active_mask
    return int(matches.sum().item()), int(active_mask.sum().item())


def _token_kind_id(token: object) -> int:
    match token:
        case NoteToken():
            return TokenKindId.NOTE
        case RestToken():
            return TokenKindId.REST
        case HoldToken():
            return TokenKindId.HOLD
        case BarToken():
            return TokenKindId.BAR
        case EndToken():
            return TokenKindId.END
        case HandToken():
            return TokenKindId.HAND
        case JoinWithPreviousToken():
            return TokenKindId.JOIN_WITH_PREVIOUS
        case StartToken():
            return TokenKindId.START
        case _:
            raise ValueError(f"unsupported token type: {type(token).__name__}")
