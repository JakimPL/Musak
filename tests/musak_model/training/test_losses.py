from typing import Final

import torch

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
from musak_model.training.config import EventObjectiveConfig
from musak_model.training.dataset.factorized import TokenAttributeTargetTensors
from musak_model.training.losses import factorized_event_loss

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
