from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from musak_model.auxiliary.schema import (
    MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
    MusicalAuxiliaryLogits,
    MusicalAuxiliaryTargetTensors,
)
from musak_model.conditioning.harmony.relations import (
    HARMONIC_RELATION_CLASS_COUNT,
    HARMONIC_RELATION_IGNORE_ID,
    HarmonicRelationTargetTensors,
)
from musak_model.conditioning.harmony.schema import HarmonicPlanInputTensors, HarmonicSlotRole
from musak_model.conditioning.harmony.vocabulary import (
    HARMONIC_PLAN_UNKNOWN_ID,
    PLAN_CONFIDENCE_VOCABULARY_SIZE,
    slot_role_to_id,
)
from musak_model.model.output import FactorizedTokenLogits
from musak_model.tokens.factorized import ABSENT_ATTRIBUTE_ID, hand_to_attribute_id
from musak_model.tokens.schema import Hand
from musak_model.training.config import (
    EventObjectiveConfig,
    HarmonicRelationObjectiveConfig,
    MusicalAuxiliaryObjectiveConfig,
)
from musak_model.training.dataset.factorized import TokenAttributeTargetTensors


@dataclass(frozen=True)
class FactorizedEventLoss:
    loss: Tensor
    kind_loss: Tensor
    duration_loss: Tensor
    degree_loss: Tensor
    accidental_loss: Tensor
    octave_offset_loss: Tensor
    hand_loss: Tensor
    kind_target_count: int
    duration_target_count: int
    degree_target_count: int
    accidental_target_count: int
    octave_offset_target_count: int
    hand_target_count: int


@dataclass(frozen=True)
class MusicalAuxiliaryLoss:
    loss: Tensor
    note_density_loss: Tensor
    rhythmic_diversity_loss: Tensor
    voice_independence_loss: Tensor
    uses_accidentals_loss: Tensor
    dotted_duration_loss: Tensor
    hand_span_loss: Tensor
    bar_note_density_loss: Tensor
    bar_rhythmic_diversity_loss: Tensor
    bar_voice_independence_loss: Tensor
    bar_uses_accidentals_loss: Tensor
    bar_dotted_duration_loss: Tensor
    bar_hand_span_loss: Tensor
    note_density_match_count: int
    note_density_target_count: int
    rhythmic_diversity_match_count: int
    rhythmic_diversity_target_count: int
    voice_independence_match_count: int
    voice_independence_target_count: int
    uses_accidentals_match_count: int
    uses_accidentals_target_count: int
    dotted_duration_match_count: int
    dotted_duration_target_count: int
    hand_span_match_count: int
    hand_span_target_count: int
    bar_note_density_match_count: int
    bar_note_density_target_count: int
    bar_rhythmic_diversity_match_count: int
    bar_rhythmic_diversity_target_count: int
    bar_voice_independence_match_count: int
    bar_voice_independence_target_count: int
    bar_uses_accidentals_match_count: int
    bar_uses_accidentals_target_count: int
    bar_dotted_duration_match_count: int
    bar_dotted_duration_target_count: int
    bar_hand_span_match_count: int
    bar_hand_span_target_count: int


@dataclass(frozen=True)
class HarmonicRelationLoss:
    loss: Tensor
    match_count: int
    target_count: int
    macro_f1: float
    target_counts: tuple[int, ...]
    prediction_counts: tuple[int, ...]


def factorized_event_loss(
    logits: FactorizedTokenLogits,
    *,
    targets: TokenAttributeTargetTensors,
    config: EventObjectiveConfig,
) -> FactorizedEventLoss:
    kind_loss, kind_target_count = _masked_cross_entropy(logits.kind, targets.kind_ids)
    duration_loss, duration_target_count = _masked_cross_entropy(logits.duration, targets.duration_ids)
    degree_loss, degree_target_count = _masked_cross_entropy(logits.degree, targets.degree_ids)
    accidental_loss, accidental_target_count = _masked_cross_entropy(logits.accidental, targets.accidental_ids)
    octave_offset_loss, octave_offset_target_count = _masked_cross_entropy(
        logits.octave_offset,
        targets.octave_offset_ids,
    )
    hand_loss, hand_target_count = _masked_cross_entropy(logits.hand, targets.hand_ids)
    loss = (
        config.kind_weight * kind_loss
        + config.duration_weight * duration_loss
        + config.degree_weight * degree_loss
        + config.accidental_weight * accidental_loss
        + config.octave_offset_weight * octave_offset_loss
        + config.hand_weight * hand_loss
    )
    return FactorizedEventLoss(
        loss=loss,
        kind_loss=kind_loss,
        duration_loss=duration_loss,
        degree_loss=degree_loss,
        accidental_loss=accidental_loss,
        octave_offset_loss=octave_offset_loss,
        hand_loss=hand_loss,
        kind_target_count=kind_target_count,
        duration_target_count=duration_target_count,
        degree_target_count=degree_target_count,
        accidental_target_count=accidental_target_count,
        octave_offset_target_count=octave_offset_target_count,
        hand_target_count=hand_target_count,
    )


def musical_auxiliary_loss(
    logits: MusicalAuxiliaryLogits,
    *,
    targets: MusicalAuxiliaryTargetTensors,
    config: MusicalAuxiliaryObjectiveConfig,
) -> MusicalAuxiliaryLoss:
    note_density_loss, note_density_match_count, note_density_target_count = _target_cross_entropy(
        logits.note_density,
        targets.note_density_ids,
    )
    rhythmic_diversity_loss, rhythmic_diversity_match_count, rhythmic_diversity_target_count = _target_cross_entropy(
        logits.rhythmic_diversity,
        targets.rhythmic_diversity_ids,
    )
    voice_independence_loss, voice_independence_match_count, voice_independence_target_count = _target_cross_entropy(
        logits.voice_independence,
        targets.voice_independence_ids,
    )
    uses_accidentals_loss, uses_accidentals_match_count, uses_accidentals_target_count = _target_cross_entropy(
        logits.uses_accidentals,
        targets.uses_accidentals_ids,
    )
    dotted_duration_loss, dotted_duration_match_count, dotted_duration_target_count = _target_cross_entropy(
        logits.dotted_duration,
        targets.dotted_duration_ids,
    )
    hand_span_loss, hand_span_match_count, hand_span_target_count = _target_cross_entropy(
        logits.hand_span,
        targets.hand_span_ids,
    )
    bar_note_density_loss, bar_note_density_match_count, bar_note_density_target_count = _target_cross_entropy(
        logits.bar.note_density,
        targets.bar_targets.note_density_ids,
    )
    bar_rhythmic_diversity_loss, bar_rhythmic_diversity_match_count, bar_rhythmic_diversity_target_count = (
        _target_cross_entropy(
            logits.bar.rhythmic_diversity,
            targets.bar_targets.rhythmic_diversity_ids,
        )
    )
    bar_voice_independence_loss, bar_voice_independence_match_count, bar_voice_independence_target_count = (
        _target_cross_entropy(
            logits.bar.voice_independence,
            targets.bar_targets.voice_independence_ids,
        )
    )
    bar_uses_accidentals_loss, bar_uses_accidentals_match_count, bar_uses_accidentals_target_count = (
        _target_cross_entropy(
            logits.bar.uses_accidentals,
            targets.bar_targets.uses_accidentals_ids,
        )
    )
    bar_dotted_duration_loss, bar_dotted_duration_match_count, bar_dotted_duration_target_count = _target_cross_entropy(
        logits.bar.dotted_duration,
        targets.bar_targets.dotted_duration_ids,
    )
    bar_hand_span_loss, bar_hand_span_match_count, bar_hand_span_target_count = _target_cross_entropy(
        logits.bar.hand_span,
        targets.bar_targets.hand_span_ids,
    )
    sequence_loss = (
        config.note_density_weight * note_density_loss
        + config.rhythmic_diversity_weight * rhythmic_diversity_loss
        + config.voice_independence_weight * voice_independence_loss
        + config.uses_accidentals_weight * uses_accidentals_loss
        + config.dotted_duration_weight * dotted_duration_loss
        + config.hand_span_weight * hand_span_loss
    )
    bar_loss = (
        config.note_density_weight * bar_note_density_loss
        + config.rhythmic_diversity_weight * bar_rhythmic_diversity_loss
        + config.voice_independence_weight * bar_voice_independence_loss
        + config.uses_accidentals_weight * bar_uses_accidentals_loss
        + config.dotted_duration_weight * bar_dotted_duration_loss
        + config.hand_span_weight * bar_hand_span_loss
    )
    return MusicalAuxiliaryLoss(
        loss=sequence_loss + config.bar_weight * bar_loss,
        note_density_loss=note_density_loss,
        rhythmic_diversity_loss=rhythmic_diversity_loss,
        voice_independence_loss=voice_independence_loss,
        uses_accidentals_loss=uses_accidentals_loss,
        dotted_duration_loss=dotted_duration_loss,
        hand_span_loss=hand_span_loss,
        bar_note_density_loss=bar_note_density_loss,
        bar_rhythmic_diversity_loss=bar_rhythmic_diversity_loss,
        bar_voice_independence_loss=bar_voice_independence_loss,
        bar_uses_accidentals_loss=bar_uses_accidentals_loss,
        bar_dotted_duration_loss=bar_dotted_duration_loss,
        bar_hand_span_loss=bar_hand_span_loss,
        note_density_match_count=note_density_match_count,
        note_density_target_count=note_density_target_count,
        rhythmic_diversity_match_count=rhythmic_diversity_match_count,
        rhythmic_diversity_target_count=rhythmic_diversity_target_count,
        voice_independence_match_count=voice_independence_match_count,
        voice_independence_target_count=voice_independence_target_count,
        uses_accidentals_match_count=uses_accidentals_match_count,
        uses_accidentals_target_count=uses_accidentals_target_count,
        dotted_duration_match_count=dotted_duration_match_count,
        dotted_duration_target_count=dotted_duration_target_count,
        hand_span_match_count=hand_span_match_count,
        hand_span_target_count=hand_span_target_count,
        bar_note_density_match_count=bar_note_density_match_count,
        bar_note_density_target_count=bar_note_density_target_count,
        bar_rhythmic_diversity_match_count=bar_rhythmic_diversity_match_count,
        bar_rhythmic_diversity_target_count=bar_rhythmic_diversity_target_count,
        bar_voice_independence_match_count=bar_voice_independence_match_count,
        bar_voice_independence_target_count=bar_voice_independence_target_count,
        bar_uses_accidentals_match_count=bar_uses_accidentals_match_count,
        bar_uses_accidentals_target_count=bar_uses_accidentals_target_count,
        bar_dotted_duration_match_count=bar_dotted_duration_match_count,
        bar_dotted_duration_target_count=bar_dotted_duration_target_count,
        bar_hand_span_match_count=bar_hand_span_match_count,
        bar_hand_span_target_count=bar_hand_span_target_count,
    )


def harmonic_relation_loss(
    logits: Tensor,
    *,
    targets: HarmonicRelationTargetTensors,
    harmonic_plan: HarmonicPlanInputTensors,
    bar_relative_ticks: Tensor,
    bar_duration_ticks: Tensor,
    active_hand_ids: Tensor,
    config: HarmonicRelationObjectiveConfig,
) -> HarmonicRelationLoss:
    if logits.shape[:-1] != targets.relation_ids.shape:
        logits_shape = tuple(logits.shape[:-1])
        target_shape = tuple(targets.relation_ids.shape)
        raise ValueError(f"relation logits shape {logits_shape} does not match targets shape {target_shape}")

    if logits.size(-1) != HARMONIC_RELATION_CLASS_COUNT:
        raise ValueError("relation logits use an unexpected class count")

    weights = _harmonic_relation_weights(
        targets,
        harmonic_plan=harmonic_plan,
        bar_relative_ticks=bar_relative_ticks,
        bar_duration_ticks=bar_duration_ticks,
        active_hand_ids=active_hand_ids,
        config=config,
    )
    flat_targets = targets.relation_ids.reshape(-1)
    active_mask = flat_targets != HARMONIC_RELATION_IGNORE_ID
    target_count = int(active_mask.sum().item())
    if target_count == 0:
        zero = logits.sum() * 0.0
        empty_counts = (0,) * HARMONIC_RELATION_CLASS_COUNT
        return HarmonicRelationLoss(
            loss=zero,
            match_count=0,
            target_count=0,
            macro_f1=0.0,
            target_counts=empty_counts,
            prediction_counts=empty_counts,
        )

    active_targets = flat_targets[active_mask]
    if torch.any(active_targets < 0):
        raise ValueError("active harmonic relation targets contain negative ids")

    if torch.any(active_targets >= HARMONIC_RELATION_CLASS_COUNT):
        raise ValueError("active harmonic relation targets contain ids outside the relation class range")

    flat_logits = logits.reshape(-1, logits.size(-1))
    active_logits = flat_logits[active_mask]
    active_weights = weights.reshape(-1)[active_mask].to(device=logits.device, dtype=logits.dtype)
    token_losses = nn.functional.cross_entropy(active_logits, active_targets, reduction="none")
    loss = (token_losses * active_weights).sum() / active_weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
    predictions = active_logits.argmax(dim=-1)
    match_count = int((predictions == active_targets).sum().item())
    return HarmonicRelationLoss(
        loss=loss,
        match_count=match_count,
        target_count=target_count,
        macro_f1=_macro_f1(predictions, active_targets, class_count=HARMONIC_RELATION_CLASS_COUNT),
        target_counts=_class_counts(active_targets, class_count=HARMONIC_RELATION_CLASS_COUNT),
        prediction_counts=_class_counts(predictions, class_count=HARMONIC_RELATION_CLASS_COUNT),
    )


def _masked_cross_entropy(logits: Tensor, targets: Tensor) -> tuple[Tensor, int]:
    if logits.shape[:-1] != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape[:-1])} does not match targets shape {tuple(targets.shape)}")

    flat_targets = targets.reshape(-1)
    active_mask = flat_targets != ABSENT_ATTRIBUTE_ID
    target_count = int(active_mask.sum().item())
    if target_count == 0:
        return logits.sum() * 0.0, 0

    flat_logits = logits.reshape(-1, logits.size(-1))
    active_targets = flat_targets[active_mask]
    if torch.any(active_targets < 0):
        raise ValueError("active factorized targets contain negative ids")

    if torch.any(active_targets >= logits.size(-1)):
        raise ValueError("active factorized targets contain ids outside the corresponding head range")

    return nn.functional.cross_entropy(flat_logits[active_mask], active_targets, reduction="mean"), target_count


def _target_cross_entropy(logits: Tensor, targets: Tensor) -> tuple[Tensor, int, int]:
    if logits.shape[:-1] != targets.shape:
        logits_shape = tuple(logits.shape[:-1])
        targets_shape = tuple(targets.shape)
        raise ValueError(f"auxiliary logits shape {logits_shape} does not match targets shape {targets_shape}")

    flat_targets = targets.reshape(-1)
    active_mask = flat_targets != MUSICAL_AUXILIARY_TARGET_IGNORE_ID
    target_count = int(active_mask.sum().item())
    if target_count == 0:
        return logits.sum() * 0.0, 0, 0

    active_targets = flat_targets[active_mask]
    if torch.any(active_targets < 0):
        raise ValueError("active auxiliary targets contain negative ids")

    if torch.any(active_targets >= logits.size(-1)):
        raise ValueError("active auxiliary targets contain ids outside the corresponding head range")

    flat_logits = logits.reshape(-1, logits.size(-1))
    active_logits = flat_logits[active_mask]
    predictions = active_logits.argmax(dim=-1)
    match_count = int((predictions == active_targets).sum().item())
    return nn.functional.cross_entropy(active_logits, active_targets, reduction="mean"), match_count, target_count


def _harmonic_relation_weights(
    targets: HarmonicRelationTargetTensors,
    *,
    harmonic_plan: HarmonicPlanInputTensors,
    bar_relative_ticks: Tensor,
    bar_duration_ticks: Tensor,
    active_hand_ids: Tensor,
    config: HarmonicRelationObjectiveConfig,
) -> Tensor:
    weights = torch.ones_like(targets.relation_ids, dtype=torch.float)
    weights = weights * _beat_weights(
        bar_relative_ticks,
        bar_duration_ticks=bar_duration_ticks,
        config=config,
    )
    weights = weights * _hand_weights(active_hand_ids, config=config)
    weights = weights * _slot_role_weights(harmonic_plan.slot_role_ids, config=config)
    weights = weights * _plan_confidence_weights(harmonic_plan.plan_confidence_ids, config=config)
    active_mask = targets.relation_ids != HARMONIC_RELATION_IGNORE_ID
    return weights * active_mask.to(dtype=weights.dtype)


def _beat_weights(
    bar_relative_ticks: Tensor,
    *,
    bar_duration_ticks: Tensor,
    config: HarmonicRelationObjectiveConfig,
) -> Tensor:
    downbeat = bar_relative_ticks == 0
    middle_of_bar = bar_relative_ticks * 2 == bar_duration_ticks.clamp_min(1)
    weights = torch.full_like(bar_relative_ticks, config.weak_beat_weight, dtype=torch.float)
    weights = torch.where(middle_of_bar, torch.full_like(weights, config.strong_beat_weight), weights)
    return torch.where(downbeat, torch.full_like(weights, config.downbeat_weight), weights)


def _hand_weights(active_hand_ids: Tensor, *, config: HarmonicRelationObjectiveConfig) -> Tensor:
    weights = torch.ones_like(active_hand_ids, dtype=torch.float)
    for hand, weight in ((Hand.RIGHT, config.right_hand_weight), (Hand.LEFT, config.left_hand_weight)):
        weights = torch.where(
            active_hand_ids == _hand_id(hand),
            torch.full_like(weights, weight),
            weights,
        )

    return weights


def _slot_role_weights(slot_role_ids: Tensor, *, config: HarmonicRelationObjectiveConfig) -> Tensor:
    weights = torch.ones_like(slot_role_ids, dtype=torch.float)
    for slot_role, weight in _slot_role_weight_pairs(config):
        weights = torch.where(
            slot_role_ids == _slot_role_id(slot_role),
            torch.full_like(weights, weight),
            weights,
        )

    return weights


def _plan_confidence_weights(plan_confidence_ids: Tensor, *, config: HarmonicRelationObjectiveConfig) -> Tensor:
    if not config.use_plan_confidence_weight:
        return torch.ones_like(plan_confidence_ids, dtype=torch.float)

    denominator = max(PLAN_CONFIDENCE_VOCABULARY_SIZE - 2, 1)
    known_weights = plan_confidence_ids.clamp(min=0, max=denominator).to(dtype=torch.float) / denominator
    weights = torch.where(
        plan_confidence_ids == HARMONIC_PLAN_UNKNOWN_ID,
        torch.full_like(known_weights, config.minimum_plan_confidence_weight),
        known_weights,
    )
    return weights.clamp_min(config.minimum_plan_confidence_weight)


def _slot_role_weight_pairs(
    config: HarmonicRelationObjectiveConfig,
) -> tuple[tuple[HarmonicSlotRole, float], ...]:
    return (
        (HarmonicSlotRole.OPENING, config.opening_weight),
        (HarmonicSlotRole.CONTINUATION, config.continuation_weight),
        (HarmonicSlotRole.CADENCE_PREPARATION, config.cadence_preparation_weight),
        (HarmonicSlotRole.CADENCE, config.cadence_weight),
    )


def _macro_f1(predictions: Tensor, targets: Tensor, *, class_count: int) -> float:
    scores: list[float] = []
    for class_id in range(class_count):
        predicted = predictions == class_id
        actual = targets == class_id
        true_positive = int((predicted & actual).sum().item())
        false_positive = int((predicted & ~actual).sum().item())
        false_negative = int((~predicted & actual).sum().item())
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator > 0:
            scores.append(2 * true_positive / denominator)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def _class_counts(values: Tensor, *, class_count: int) -> tuple[int, ...]:
    counts = torch.bincount(values, minlength=class_count)
    return tuple(int(count.item()) for count in counts[:class_count])


def _hand_id(hand: Hand) -> int:
    return hand_to_attribute_id(hand)


def _slot_role_id(slot_role: HarmonicSlotRole) -> int:
    return slot_role_to_id(slot_role)
