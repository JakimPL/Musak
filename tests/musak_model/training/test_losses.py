from typing import Final

import torch
import torch.nn.functional as functional
from pytest import approx

from musak_model.auxiliary.schema import (
    MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
    MusicalAuxiliaryLogits,
    MusicalAuxiliaryTargetTensors,
    MusicalBarAuxiliaryLogits,
    MusicalBarAuxiliaryTargetTensors,
)
from musak_model.conditioning.harmony.fields import HARMONIC_PLAN_TENSOR_FIELDS
from musak_model.conditioning.harmony.reconstruction import HARMONIC_PLAN_RECONSTRUCTION_FIELDS
from musak_model.conditioning.harmony.relations import (
    HARMONIC_RELATION_CLASS_COUNT,
    HARMONIC_RELATION_IGNORE_ID,
    HarmonicRelationId,
    HarmonicRelationTargetTensors,
)
from musak_model.conditioning.harmony.schema import HarmonicPlanInputTensors, HarmonicSlotRole
from musak_model.conditioning.harmony.vocabulary import plan_confidence_to_id, slot_role_to_id
from musak_model.model.config import ModelOutputMode
from musak_model.model.output import (
    FactorizedTokenLogits,
    HarmonicPlanContrastiveEmbeddings,
    HarmonicPlanReconstructionLogits,
)
from musak_model.tokens.factorized import (
    ABSENT_ATTRIBUTE_ID,
    ACCIDENTAL_ATTRIBUTE_COUNT,
    DEGREE_ATTRIBUTE_COUNT,
    HAND_ATTRIBUTE_COUNT,
    OCTAVE_OFFSET_ATTRIBUTE_COUNT,
    TOKEN_KIND_COUNT,
    hand_to_attribute_id,
)
from musak_model.tokens.schema import Hand
from musak_model.training.config import (
    EventObjectiveConfig,
    HarmonicPlanContrastiveObjectiveConfig,
    HarmonicPlanReconstructionObjectiveConfig,
    HarmonicRelationObjectiveConfig,
    MusicalAuxiliaryObjectiveConfig,
)
from musak_model.training.dataset.factorized import TokenAttributeTargetTensors
from musak_model.training.losses import (
    factorized_event_loss,
    harmonic_plan_contrastive_loss,
    harmonic_plan_reconstruction_loss,
    harmonic_relation_loss,
    musical_auxiliary_loss,
)

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
        bar_weight=1.0,
        note_density_weight=1.0,
        rhythmic_diversity_weight=1.0,
        voice_independence_weight=1.0,
        uses_accidentals_weight=1.0,
        dotted_duration_weight=1.0,
        hand_span_weight=1.0,
    )


def _harmonic_relation_objective_config() -> HarmonicRelationObjectiveConfig:
    return HarmonicRelationObjectiveConfig(
        enabled=True,
        weight=0.03,
        downbeat_weight=2.0,
        strong_beat_weight=1.0,
        weak_beat_weight=0.5,
        left_hand_weight=1.0,
        right_hand_weight=1.0,
        opening_weight=1.0,
        continuation_weight=1.0,
        cadence_preparation_weight=1.0,
        cadence_weight=3.0,
        use_plan_confidence_weight=True,
        minimum_plan_confidence_weight=0.5,
    )


def _harmonic_plan_reconstruction_objective_config() -> HarmonicPlanReconstructionObjectiveConfig:
    return HarmonicPlanReconstructionObjectiveConfig(
        enabled=True,
        weight=0.02,
        harmonic_function_weight=1.0,
        root_degree_weight=1.0,
        quality_weight=0.5,
        extension_weight=0.25,
        cadence_strength_weight=0.5,
    )


def _harmonic_plan_contrastive_objective_config() -> HarmonicPlanContrastiveObjectiveConfig:
    return HarmonicPlanContrastiveObjectiveConfig(
        enabled=True,
        weight=0.01,
        negative_count=1,
        temperature=0.20,
    )


def _harmonic_plan(slot_role_ids: torch.Tensor, plan_confidence_ids: torch.Tensor) -> HarmonicPlanInputTensors:
    ids = torch.zeros_like(slot_role_ids)
    values = {field.name: ids for field in HARMONIC_PLAN_TENSOR_FIELDS}
    values["slot_role_ids"] = slot_role_ids
    values["plan_confidence_ids"] = plan_confidence_ids
    return HarmonicPlanInputTensors(**values)


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
        bar=MusicalBarAuxiliaryLogits(
            note_density=torch.tensor([[[0.0, 3.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]),
            rhythmic_diversity=torch.tensor([[[0.0, 3.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]),
            voice_independence=torch.tensor([[[0.0, 3.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]),
            uses_accidentals=torch.tensor([[[0.0, 3.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]),
            dotted_duration=torch.tensor([[[0.0, 3.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]),
            hand_span=torch.tensor([[[0.0, 3.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]),
        ),
    )
    targets = MusicalAuxiliaryTargetTensors(
        note_density_ids=torch.tensor([1, MUSICAL_AUXILIARY_TARGET_IGNORE_ID]),
        rhythmic_diversity_ids=torch.tensor([1, 0]),
        voice_independence_ids=torch.tensor([1, 1]),
        uses_accidentals_ids=torch.tensor([1, 0]),
        dotted_duration_ids=torch.tensor([1, 0]),
        hand_span_ids=torch.tensor([1, 1]),
        bar_targets=MusicalBarAuxiliaryTargetTensors(
            note_density_ids=torch.tensor([[1, MUSICAL_AUXILIARY_TARGET_IGNORE_ID], [0, 1]]),
            rhythmic_diversity_ids=torch.tensor([[1, 0], [0, 1]]),
            voice_independence_ids=torch.tensor([[1, 0], [0, 1]]),
            uses_accidentals_ids=torch.tensor([[1, 0], [0, 1]]),
            dotted_duration_ids=torch.tensor([[1, 0], [0, 1]]),
            hand_span_ids=torch.tensor([[1, 0], [0, 1]]),
        ),
    )

    loss = musical_auxiliary_loss(logits, targets=targets, config=_auxiliary_objective_config())

    assert loss.note_density_target_count == 1
    assert loss.note_density_match_count == 1
    assert loss.rhythmic_diversity_target_count == 2
    assert loss.rhythmic_diversity_match_count == 2
    assert loss.voice_independence_match_count == 2
    assert loss.bar_note_density_target_count == 3
    assert loss.bar_note_density_match_count == 3
    assert loss.bar_rhythmic_diversity_target_count == 4
    assert loss.bar_rhythmic_diversity_match_count == 4
    assert loss.loss.item() > 0.0


def test_harmonic_relation_loss_weights_active_note_targets() -> None:
    logits = torch.tensor(
        [
            [
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 2.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    targets = HarmonicRelationTargetTensors(
        relation_ids=torch.tensor(
            [
                [
                    HarmonicRelationId.CHORD_ROOT,
                    HarmonicRelationId.CHORD_THIRD,
                    HarmonicRelationId.CHORD_FIFTH,
                    HARMONIC_RELATION_IGNORE_ID,
                ]
            ],
            dtype=torch.long,
        )
    )
    harmonic_plan = _harmonic_plan(
        torch.tensor(
            [
                [
                    slot_role_to_id(HarmonicSlotRole.OPENING),
                    slot_role_to_id(HarmonicSlotRole.CADENCE),
                    slot_role_to_id(HarmonicSlotRole.CONTINUATION),
                    slot_role_to_id(HarmonicSlotRole.CADENCE),
                ]
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [
                [
                    plan_confidence_to_id(1.0),
                    plan_confidence_to_id(1.0),
                    plan_confidence_to_id(1.0),
                    plan_confidence_to_id(1.0),
                ]
            ],
            dtype=torch.long,
        ),
    )

    loss = harmonic_relation_loss(
        logits,
        targets=targets,
        harmonic_plan=harmonic_plan,
        bar_relative_ticks=torch.tensor([[0, 2, 1, 0]]),
        bar_duration_ticks=torch.tensor([[4, 4, 4, 4]]),
        active_hand_ids=torch.tensor(
            [
                [
                    hand_to_attribute_id(Hand.RIGHT),
                    hand_to_attribute_id(Hand.LEFT),
                    hand_to_attribute_id(Hand.RIGHT),
                    hand_to_attribute_id(Hand.RIGHT),
                ]
            ]
        ),
        config=_harmonic_relation_objective_config(),
    )

    token_losses = functional.cross_entropy(logits[0, :3], targets.relation_ids[0, :3], reduction="none")
    expected_weights = torch.tensor([2.0, 3.0, 0.5])
    expected_loss = (token_losses * expected_weights).sum() / expected_weights.sum()
    assert loss.loss.item() == approx(expected_loss.item())
    assert loss.match_count == 2
    assert loss.target_count == 3
    assert loss.target_counts == (1, 1, 1, 0, 0, 0, 0)
    assert loss.prediction_counts == (1, 0, 2, 0, 0, 0, 0)
    assert len(loss.target_counts) == HARMONIC_RELATION_CLASS_COUNT


def test_harmonic_plan_reconstruction_loss_pools_by_known_harmonic_slot() -> None:
    target_ids = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    values = {field.name: target_ids.clone() for field in HARMONIC_PLAN_TENSOR_FIELDS}
    values["remaining_harmonic_slot_ids"] = torch.tensor([[2, 2, 1, 0]], dtype=torch.long)
    harmonic_plan = HarmonicPlanInputTensors(**values)
    logits_by_field = {
        field.name: torch.zeros(1, 4, field.vocabulary_size) for field in HARMONIC_PLAN_RECONSTRUCTION_FIELDS
    }
    for field_logits in logits_by_field.values():
        field_logits[0, 0, 1] = 4.0
        field_logits[0, 1, 1] = 2.0
        field_logits[0, 2, 1] = 4.0
        field_logits[0, 3, 0] = 4.0

    loss = harmonic_plan_reconstruction_loss(
        HarmonicPlanReconstructionLogits(logits_by_field=logits_by_field),
        harmonic_plan=harmonic_plan,
        token_padding_mask=torch.zeros(1, 4, dtype=torch.bool),
        config=_harmonic_plan_reconstruction_objective_config(),
    )

    assert loss.loss.item() > 0.0
    for field in HARMONIC_PLAN_RECONSTRUCTION_FIELDS:
        assert loss.field_match_counts[field.name] == 2
        assert loss.field_target_counts[field.name] == 2


def test_harmonic_plan_contrastive_loss_uses_in_batch_negatives() -> None:
    embeddings = HarmonicPlanContrastiveEmbeddings(
        music_embeddings=torch.eye(3),
        plan_embeddings=torch.eye(3),
    )

    loss = harmonic_plan_contrastive_loss(
        embeddings,
        config=_harmonic_plan_contrastive_objective_config(),
    )

    assert loss.target_count == 3
    assert loss.match_count == 3
    assert loss.positive_similarity == approx(1.0)
    assert loss.negative_similarity == approx(0.0)
    assert loss.loss.item() < 0.01


def test_harmonic_plan_contrastive_loss_ignores_single_sample_batch() -> None:
    embeddings = HarmonicPlanContrastiveEmbeddings(
        music_embeddings=torch.ones(1, 3),
        plan_embeddings=torch.ones(1, 3),
    )

    loss = harmonic_plan_contrastive_loss(
        embeddings,
        config=_harmonic_plan_contrastive_objective_config(),
    )

    assert loss.loss.item() == 0.0
    assert loss.target_count == 0
