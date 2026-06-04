from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from musak_model.auxiliary.schema import MusicalAuxiliaryLogits
from musak_model.tokens.factorized import ABSENT_ATTRIBUTE_ID, TokenAttributes


@dataclass(frozen=True)
class FactorizedTokenLogits:
    kind: Tensor
    degree: Tensor
    accidental: Tensor
    octave_offset: Tensor
    duration: Tensor
    hand: Tensor


@dataclass(frozen=True)
class ModelTrainingLogits:
    flat_logits: Tensor
    musical_auxiliary_logits: MusicalAuxiliaryLogits
    factorized_logits: FactorizedTokenLogits | None = None
    harmonic_relation_logits: Tensor | None = None
    harmony_gate_values: Tensor | None = None


@dataclass(frozen=True)
class FlatTokenAttributeBuffers:
    kind_ids: Tensor
    degree_ids: Tensor
    accidental_ids: Tensor
    octave_offset_ids: Tensor
    duration_ids: Tensor
    hand_ids: Tensor

    @classmethod
    def from_attributes(cls, attributes: tuple[TokenAttributes, ...]) -> FlatTokenAttributeBuffers:
        return cls(
            kind_ids=_tensor_from_values([attribute.kind_id for attribute in attributes]),
            degree_ids=_tensor_from_values([attribute.degree_id for attribute in attributes]),
            accidental_ids=_tensor_from_values([attribute.accidental_id for attribute in attributes]),
            octave_offset_ids=_tensor_from_values([attribute.octave_offset_id for attribute in attributes]),
            duration_ids=_tensor_from_values([attribute.duration_id for attribute in attributes]),
            hand_ids=_tensor_from_values([attribute.hand_id for attribute in attributes]),
        )


@dataclass(frozen=True)
class _FlatAttributeScoreInput:
    logits: Tensor
    attribute_ids: Tensor


def flat_token_log_scores(
    logits: FactorizedTokenLogits,
    *,
    flat_attributes: FlatTokenAttributeBuffers,
) -> Tensor:
    scores = _flat_attribute_scores(
        _FlatAttributeScoreInput(
            logits=logits.kind,
            attribute_ids=flat_attributes.kind_ids,
        )
    )
    for score_input in _optional_flat_attribute_score_inputs(logits, flat_attributes):
        scores = scores + _optional_flat_attribute_scores(score_input)

    return scores


def _optional_flat_attribute_score_inputs(
    logits: FactorizedTokenLogits,
    flat_attributes: FlatTokenAttributeBuffers,
) -> tuple[_FlatAttributeScoreInput, ...]:
    return (
        _FlatAttributeScoreInput(logits.degree, flat_attributes.degree_ids),
        _FlatAttributeScoreInput(logits.accidental, flat_attributes.accidental_ids),
        _FlatAttributeScoreInput(logits.octave_offset, flat_attributes.octave_offset_ids),
        _FlatAttributeScoreInput(logits.duration, flat_attributes.duration_ids),
        _FlatAttributeScoreInput(logits.hand, flat_attributes.hand_ids),
    )


def _optional_flat_attribute_scores(score_input: _FlatAttributeScoreInput) -> Tensor:
    safe_attribute_ids = score_input.attribute_ids.clamp_min(0)
    scores = _flat_attribute_scores(
        _FlatAttributeScoreInput(
            logits=score_input.logits,
            attribute_ids=safe_attribute_ids,
        )
    )
    active_mask = score_input.attribute_ids != ABSENT_ATTRIBUTE_ID
    return scores * active_mask.to(device=scores.device, dtype=scores.dtype).view(1, 1, -1)


def _flat_attribute_scores(score_input: _FlatAttributeScoreInput) -> Tensor:
    log_probabilities = torch.log_softmax(score_input.logits, dim=-1)
    return _gather_flat_attribute_scores(log_probabilities, score_input.attribute_ids)


def _gather_flat_attribute_scores(log_probabilities: Tensor, attribute_ids: Tensor) -> Tensor:
    return log_probabilities.index_select(dim=-1, index=attribute_ids.to(log_probabilities.device))


def _tensor_from_values(values: list[int]) -> Tensor:
    return torch.tensor(values, dtype=torch.long)
