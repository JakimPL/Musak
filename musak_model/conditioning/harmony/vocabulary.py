from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypeVar

import torch
from torch import Tensor

from musak_model.conditioning.buckets import (
    optional_float_threshold_bucket_id,
    optional_integer_threshold_bucket_id,
    threshold_bucket_vocabulary_size,
)
from musak_model.conditioning.harmony.schema import (
    HarmonicPlanIds,
    HarmonicPlanInputTensors,
    HarmonicPlanWindow,
    HarmonicSlotRole,
    harmonic_function_for_chord,
)
from musak_model.harmony.schema import Chord, ChordExtension, ChordQuality
from musak_model.tokens.schema import (
    MAX_ACCIDENTAL,
    MAX_DEGREE,
    MIN_ACCIDENTAL,
    MIN_DEGREE,
)
from musak_shared.elements import HarmonicFunction

HARMONIC_PLAN_UNKNOWN_ID: Final[int] = 0
_HARMONIC_PLAN_ACTIVE_ID_OFFSET: Final[int] = 1

_ValueT = TypeVar("_ValueT")


def _optional_vocabulary_size(order: tuple[_ValueT, ...]) -> int:
    return len(order) + _HARMONIC_PLAN_ACTIVE_ID_OFFSET


_HARMONIC_FUNCTION_ORDER: Final[tuple[HarmonicFunction, ...]] = tuple(HarmonicFunction)
_ROOT_DEGREE_ORDER: Final[tuple[int, ...]] = tuple(range(MIN_DEGREE, MAX_DEGREE + 1))
_ROOT_ACCIDENTAL_ORDER: Final[tuple[int, ...]] = tuple(range(MIN_ACCIDENTAL, MAX_ACCIDENTAL + 1))
_CHORD_QUALITY_ORDER: Final[tuple[ChordQuality, ...]] = tuple(ChordQuality)
_CHORD_EXTENSION_ORDER: Final[tuple[ChordExtension, ...]] = tuple(ChordExtension)
_CHORD_CHANGE_ORDER: Final[tuple[bool, ...]] = (False, True)
_SLOT_ROLE_ORDER: Final[tuple[HarmonicSlotRole, ...]] = tuple(HarmonicSlotRole)
_COUNT_BUCKET_THRESHOLDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
_REMAINING_BAR_BUCKET_THRESHOLDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
_NORMALIZED_FIELD_BUCKET_THRESHOLDS: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75, 1.0)

HARMONIC_FUNCTION_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_HARMONIC_FUNCTION_ORDER)
ROOT_DEGREE_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_ROOT_DEGREE_ORDER)
ROOT_ACCIDENTAL_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_ROOT_ACCIDENTAL_ORDER)
CHORD_QUALITY_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_CHORD_QUALITY_ORDER)
CHORD_EXTENSION_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_CHORD_EXTENSION_ORDER)
CHORD_CHANGE_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_CHORD_CHANGE_ORDER)
SLOT_ROLE_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_SLOT_ROLE_ORDER)
DISTANCE_TO_END_VOCABULARY_SIZE: Final[int] = threshold_bucket_vocabulary_size(_COUNT_BUCKET_THRESHOLDS)
CADENCE_STRENGTH_VOCABULARY_SIZE: Final[int] = threshold_bucket_vocabulary_size(_NORMALIZED_FIELD_BUCKET_THRESHOLDS)
TENSION_LEVEL_VOCABULARY_SIZE: Final[int] = threshold_bucket_vocabulary_size(_NORMALIZED_FIELD_BUCKET_THRESHOLDS)
PLAN_CONFIDENCE_VOCABULARY_SIZE: Final[int] = threshold_bucket_vocabulary_size(_NORMALIZED_FIELD_BUCKET_THRESHOLDS)
REMAINING_BAR_VOCABULARY_SIZE: Final[int] = threshold_bucket_vocabulary_size(_REMAINING_BAR_BUCKET_THRESHOLDS)
REMAINING_HARMONIC_SLOT_VOCABULARY_SIZE: Final[int] = threshold_bucket_vocabulary_size(_COUNT_BUCKET_THRESHOLDS)


def harmonic_function_to_id(harmonic_function: HarmonicFunction | None) -> int:
    return _optional_value_to_id(
        harmonic_function,
        order=_HARMONIC_FUNCTION_ORDER,
        value_name="harmonic_function",
    )


def id_to_harmonic_function(harmonic_function_id: int) -> HarmonicFunction | None:
    return _id_to_optional_value(
        harmonic_function_id,
        order=_HARMONIC_FUNCTION_ORDER,
        value_name="harmonic_function_id",
    )


def root_degree_to_id(root_degree: int | None) -> int:
    return _optional_value_to_id(root_degree, order=_ROOT_DEGREE_ORDER, value_name="root_degree")


def id_to_root_degree(root_degree_id: int) -> int | None:
    return _id_to_optional_value(root_degree_id, order=_ROOT_DEGREE_ORDER, value_name="root_degree_id")


def root_accidental_to_id(root_accidental: int | None) -> int:
    return _optional_value_to_id(
        root_accidental,
        order=_ROOT_ACCIDENTAL_ORDER,
        value_name="root_accidental",
    )


def id_to_root_accidental(root_accidental_id: int) -> int | None:
    return _id_to_optional_value(
        root_accidental_id,
        order=_ROOT_ACCIDENTAL_ORDER,
        value_name="root_accidental_id",
    )


def chord_quality_to_id(quality: ChordQuality | None) -> int:
    return _optional_value_to_id(quality, order=_CHORD_QUALITY_ORDER, value_name="quality")


def id_to_chord_quality(quality_id: int) -> ChordQuality | None:
    return _id_to_optional_value(quality_id, order=_CHORD_QUALITY_ORDER, value_name="quality_id")


def chord_extension_to_id(extension: ChordExtension | None) -> int:
    return _optional_value_to_id(extension, order=_CHORD_EXTENSION_ORDER, value_name="extension")


def id_to_chord_extension(extension_id: int) -> ChordExtension | None:
    return _id_to_optional_value(extension_id, order=_CHORD_EXTENSION_ORDER, value_name="extension_id")


def chord_change_to_id(chord_changed: bool | None) -> int:
    return _optional_value_to_id(chord_changed, order=_CHORD_CHANGE_ORDER, value_name="chord_changed")


def id_to_chord_change(chord_change_id: int) -> bool | None:
    return _id_to_optional_value(chord_change_id, order=_CHORD_CHANGE_ORDER, value_name="chord_change_id")


def slot_role_to_id(slot_role: HarmonicSlotRole | None) -> int:
    return _optional_value_to_id(slot_role, order=_SLOT_ROLE_ORDER, value_name="slot_role")


def id_to_slot_role(slot_role_id: int) -> HarmonicSlotRole | None:
    return _id_to_optional_value(slot_role_id, order=_SLOT_ROLE_ORDER, value_name="slot_role_id")


def distance_to_end_to_id(distance_to_end: int | None) -> int:
    return _non_negative_integer_bucket_id(
        distance_to_end,
        thresholds=_COUNT_BUCKET_THRESHOLDS,
        value_name="distance_to_end",
    )


def cadence_strength_to_id(cadence_strength: float | None) -> int:
    return _non_negative_float_bucket_id(
        cadence_strength,
        thresholds=_NORMALIZED_FIELD_BUCKET_THRESHOLDS,
        value_name="cadence_strength",
    )


def tension_level_to_id(tension_level: float | None) -> int:
    return _non_negative_float_bucket_id(
        tension_level,
        thresholds=_NORMALIZED_FIELD_BUCKET_THRESHOLDS,
        value_name="tension_level",
    )


def plan_confidence_to_id(plan_confidence: float | None) -> int:
    return _non_negative_float_bucket_id(
        plan_confidence,
        thresholds=_NORMALIZED_FIELD_BUCKET_THRESHOLDS,
        value_name="plan_confidence",
    )


def remaining_bar_count_to_id(remaining_bar_count: int | None) -> int:
    return _non_negative_integer_bucket_id(
        remaining_bar_count,
        thresholds=_REMAINING_BAR_BUCKET_THRESHOLDS,
        value_name="remaining_bar_count",
    )


def remaining_harmonic_slot_count_to_id(remaining_harmonic_slot_count: int | None) -> int:
    return _non_negative_integer_bucket_id(
        remaining_harmonic_slot_count,
        thresholds=_COUNT_BUCKET_THRESHOLDS,
        value_name="remaining_harmonic_slot_count",
    )


def unknown_harmonic_plan_ids() -> HarmonicPlanIds:
    return HarmonicPlanIds(
        harmonic_function_id=HARMONIC_PLAN_UNKNOWN_ID,
        root_degree_id=HARMONIC_PLAN_UNKNOWN_ID,
        root_accidental_id=HARMONIC_PLAN_UNKNOWN_ID,
        quality_id=HARMONIC_PLAN_UNKNOWN_ID,
        extension_id=HARMONIC_PLAN_UNKNOWN_ID,
        chord_change_id=HARMONIC_PLAN_UNKNOWN_ID,
        slot_role_id=HARMONIC_PLAN_UNKNOWN_ID,
        distance_to_end_id=HARMONIC_PLAN_UNKNOWN_ID,
        cadence_strength_id=HARMONIC_PLAN_UNKNOWN_ID,
        tension_level_id=HARMONIC_PLAN_UNKNOWN_ID,
        plan_confidence_id=HARMONIC_PLAN_UNKNOWN_ID,
        remaining_bar_id=HARMONIC_PLAN_UNKNOWN_ID,
        remaining_harmonic_slot_id=HARMONIC_PLAN_UNKNOWN_ID,
    )


def harmonic_plan_ids_from_window(
    window: HarmonicPlanWindow,
    *,
    chord_changed: bool,
) -> HarmonicPlanIds:
    return HarmonicPlanIds(
        harmonic_function_id=harmonic_function_to_id(window.harmonic_function),
        root_degree_id=root_degree_to_id(window.chord.root_degree),
        root_accidental_id=root_accidental_to_id(window.chord.root_accidental),
        quality_id=chord_quality_to_id(window.chord.quality),
        extension_id=chord_extension_to_id(window.chord.extension),
        chord_change_id=chord_change_to_id(chord_changed),
        slot_role_id=slot_role_to_id(window.slot_role),
        distance_to_end_id=distance_to_end_to_id(window.distance_to_end),
        cadence_strength_id=cadence_strength_to_id(window.cadence_strength),
        tension_level_id=tension_level_to_id(window.tension_level),
        plan_confidence_id=plan_confidence_to_id(window.plan_confidence),
        remaining_bar_id=HARMONIC_PLAN_UNKNOWN_ID,
        remaining_harmonic_slot_id=remaining_harmonic_slot_count_to_id(window.distance_to_end),
    )


def harmonic_plan_ids_from_chord(
    chord: Chord,
    *,
    chord_changed: bool,
) -> HarmonicPlanIds:
    return HarmonicPlanIds(
        harmonic_function_id=harmonic_function_to_id(harmonic_function_for_chord(chord)),
        root_degree_id=root_degree_to_id(chord.root_degree),
        root_accidental_id=root_accidental_to_id(chord.root_accidental),
        quality_id=chord_quality_to_id(chord.quality),
        extension_id=chord_extension_to_id(chord.extension),
        chord_change_id=chord_change_to_id(chord_changed),
        slot_role_id=HARMONIC_PLAN_UNKNOWN_ID,
        distance_to_end_id=HARMONIC_PLAN_UNKNOWN_ID,
        cadence_strength_id=HARMONIC_PLAN_UNKNOWN_ID,
        tension_level_id=HARMONIC_PLAN_UNKNOWN_ID,
        plan_confidence_id=HARMONIC_PLAN_UNKNOWN_ID,
        remaining_bar_id=HARMONIC_PLAN_UNKNOWN_ID,
        remaining_harmonic_slot_id=HARMONIC_PLAN_UNKNOWN_ID,
    )


def harmonic_plan_ids_from_windows(windows: Sequence[HarmonicPlanWindow]) -> tuple[HarmonicPlanIds, ...]:
    ids: list[HarmonicPlanIds] = []
    previous_chord: Chord | None = None
    for window in windows:
        chord_changed = previous_chord is None or window.chord != previous_chord
        ids.append(harmonic_plan_ids_from_window(window, chord_changed=chord_changed))
        previous_chord = window.chord

    return tuple(ids)


def harmonic_plan_tensors_from_ids(ids: Sequence[HarmonicPlanIds]) -> HarmonicPlanInputTensors:
    return HarmonicPlanInputTensors(
        harmonic_function_ids=_tensor_from_values([item.harmonic_function_id for item in ids]),
        root_degree_ids=_tensor_from_values([item.root_degree_id for item in ids]),
        root_accidental_ids=_tensor_from_values([item.root_accidental_id for item in ids]),
        quality_ids=_tensor_from_values([item.quality_id for item in ids]),
        extension_ids=_tensor_from_values([item.extension_id for item in ids]),
        chord_change_ids=_tensor_from_values([item.chord_change_id for item in ids]),
        slot_role_ids=_tensor_from_values([item.slot_role_id for item in ids]),
        distance_to_end_ids=_tensor_from_values([item.distance_to_end_id for item in ids]),
        cadence_strength_ids=_tensor_from_values([item.cadence_strength_id for item in ids]),
        tension_level_ids=_tensor_from_values([item.tension_level_id for item in ids]),
        plan_confidence_ids=_tensor_from_values([item.plan_confidence_id for item in ids]),
        remaining_bar_ids=_tensor_from_values([item.remaining_bar_id for item in ids]),
        remaining_harmonic_slot_ids=_tensor_from_values([item.remaining_harmonic_slot_id for item in ids]),
    )


def _non_negative_integer_bucket_id(
    value: int | None,
    *,
    thresholds: tuple[int, ...],
    value_name: str,
) -> int:
    if value is not None and value < 0:
        raise ValueError(f"{value_name} must be non-negative")

    return optional_integer_threshold_bucket_id(value, thresholds, unknown_id=HARMONIC_PLAN_UNKNOWN_ID)


def _non_negative_float_bucket_id(
    value: float | None,
    *,
    thresholds: tuple[float, ...],
    value_name: str,
) -> int:
    if value is not None and value < 0.0:
        raise ValueError(f"{value_name} must be non-negative")

    return optional_float_threshold_bucket_id(value, thresholds, unknown_id=HARMONIC_PLAN_UNKNOWN_ID)


def _optional_value_to_id(
    value: _ValueT | None,
    *,
    order: tuple[_ValueT, ...],
    value_name: str,
) -> int:
    if value is None:
        return HARMONIC_PLAN_UNKNOWN_ID

    try:
        return order.index(value) + _HARMONIC_PLAN_ACTIVE_ID_OFFSET
    except ValueError as error:
        raise ValueError(f"unknown {value_name}: {value!r}") from error


def _id_to_optional_value(
    identifier: int,
    *,
    order: tuple[_ValueT, ...],
    value_name: str,
) -> _ValueT | None:
    if identifier == HARMONIC_PLAN_UNKNOWN_ID:
        return None

    index = identifier - _HARMONIC_PLAN_ACTIVE_ID_OFFSET
    if not 0 <= index < len(order):
        raise ValueError(f"{value_name} must be in [0, {len(order)}], got {identifier}")

    return order[index]


def _tensor_from_values(values: list[int]) -> Tensor:
    return torch.tensor(values, dtype=torch.long)
