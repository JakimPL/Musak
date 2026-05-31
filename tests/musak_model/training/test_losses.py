from typing import Final

import torch

from musak_model.auxiliary.schema import (
    MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
    MusicalAuxiliaryLogits,
    MusicalAuxiliaryTargetTensors,
)
from musak_model.model.config import ModelOutputMode
from musak_model.model.output import FactorizedTokenLogits
from musak_model.tokens.factorized import (
    ABSENT_ATTRIBUTE_ID,
    ACCIDENTAL_ATTRIBUTE_COUNT,
    DEGREE_ATTRIBUTE_COUNT,
    HAND_ATTRIBUTE_COUNT,
    OCTAVE_OFFSET_ATTRIBUTE_COUNT,
    TOKEN_KIND_COUNT,
)
from musak_model.training.config import EventObjectiveConfig, MusicalAuxiliaryObjectiveConfig
from musak_model.training.dataset.factorized import TokenAttributeTargetTensors
from musak_model.training.losses import factorized_event_loss, musical_auxiliary_loss

DURATION_ATTRIBUTE_COUNT: Final[int] = 4


def _objective_config() -> EventObjectiveConfig:
    return EventObjectiveConfig(
        mode=ModelOutputMode.FACTORIZED,
        kind_weight=1.0,
        duration_weight=1.0,
        degree_weight=1.0,
        accidental_weight=1.0,
        octave_offset_weight=1.0,
        hand_weight=1.0,
    )


def _auxiliary_objective_config() -> MusicalAuxiliaryObjectiveConfig:
    return MusicalAuxiliaryObjectiveConfig(
        enabled=True,
        weight=0.1,
        note_density_weight=1.0,
        rhythmic_diversity_weight=1.0,
        voice_independence_weight=1.0,
        uses_accidentals_weight=1.0,
        dotted_duration_weight=1.0,
        hand_span_weight=1.0,
    )


def test_factorized_event_loss_masks_inactive_attribute_targets() -> None:
    logits = FactorizedTokenLogits(
        kind=torch.zeros(1, 3, TOKEN_KIND_COUNT),
        degree=torch.zeros(1, 3, DEGREE_ATTRIBUTE_COUNT),
        accidental=torch.zeros(1, 3, ACCIDENTAL_ATTRIBUTE_COUNT),
        octave_offset=torch.zeros(1, 3, OCTAVE_OFFSET_ATTRIBUTE_COUNT),
        duration=torch.zeros(1, 3, DURATION_ATTRIBUTE_COUNT),
        hand=torch.zeros(1, 3, HAND_ATTRIBUTE_COUNT),
    )
    targets = TokenAttributeTargetTensors(
        kind_ids=torch.tensor([[0, 1, ABSENT_ATTRIBUTE_ID]]),
        degree_ids=torch.tensor([[2, ABSENT_ATTRIBUTE_ID, ABSENT_ATTRIBUTE_ID]]),
        accidental_ids=torch.tensor([[1, ABSENT_ATTRIBUTE_ID, ABSENT_ATTRIBUTE_ID]]),
        octave_offset_ids=torch.tensor([[2, ABSENT_ATTRIBUTE_ID, ABSENT_ATTRIBUTE_ID]]),
        duration_ids=torch.tensor([[3, 1, ABSENT_ATTRIBUTE_ID]]),
        hand_ids=torch.tensor([[ABSENT_ATTRIBUTE_ID, ABSENT_ATTRIBUTE_ID, ABSENT_ATTRIBUTE_ID]]),
    )

    loss = factorized_event_loss(logits, targets=targets, config=_objective_config())

    assert loss.kind_target_count == 2
    assert loss.duration_target_count == 2
    assert loss.degree_target_count == 1
    assert loss.accidental_target_count == 1
    assert loss.octave_offset_target_count == 1
    assert loss.hand_target_count == 0
    assert float(loss.hand_loss.item()) == 0.0
    assert loss.loss.item() > 0.0


def test_musical_auxiliary_loss_masks_missing_targets_and_counts_matches() -> None:
    logits = MusicalAuxiliaryLogits(
        note_density=torch.tensor([[0.0, 3.0], [2.0, 0.0]]),
        rhythmic_diversity=torch.tensor([[0.0, 4.0], [1.0, 0.0]]),
        voice_independence=torch.tensor([[0.0, 5.0], [0.0, 1.0]]),
        uses_accidentals=torch.tensor([[0.0, 6.0], [1.0, 0.0]]),
        dotted_duration=torch.tensor([[0.0, 7.0], [1.0, 0.0]]),
        hand_span=torch.tensor([[0.0, 8.0], [0.0, 1.0]]),
    )
    targets = MusicalAuxiliaryTargetTensors(
        note_density_ids=torch.tensor([1, MUSICAL_AUXILIARY_TARGET_IGNORE_ID]),
        rhythmic_diversity_ids=torch.tensor([1, 0]),
        voice_independence_ids=torch.tensor([1, 1]),
        uses_accidentals_ids=torch.tensor([1, 0]),
        dotted_duration_ids=torch.tensor([1, 0]),
        hand_span_ids=torch.tensor([1, 1]),
    )

    loss = musical_auxiliary_loss(logits, targets=targets, config=_auxiliary_objective_config())

    assert loss.note_density_target_count == 1
    assert loss.note_density_match_count == 1
    assert loss.rhythmic_diversity_target_count == 2
    assert loss.rhythmic_diversity_match_count == 2
    assert loss.voice_independence_match_count == 2
    assert loss.loss.item() > 0.0
