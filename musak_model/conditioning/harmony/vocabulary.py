from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypeVar

import torch
from torch import Tensor

from musak_model.conditioning.harmony.schema import (
    HarmonicPlanIds,
    HarmonicPlanInputTensors,
    HarmonicPlanWindow,
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

HARMONIC_FUNCTION_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_HARMONIC_FUNCTION_ORDER)
ROOT_DEGREE_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_ROOT_DEGREE_ORDER)
ROOT_ACCIDENTAL_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_ROOT_ACCIDENTAL_ORDER)
CHORD_QUALITY_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_CHORD_QUALITY_ORDER)
CHORD_EXTENSION_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_CHORD_EXTENSION_ORDER)
CHORD_CHANGE_VOCABULARY_SIZE: Final[int] = _optional_vocabulary_size(_CHORD_CHANGE_ORDER)


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


def unknown_harmonic_plan_ids() -> HarmonicPlanIds:
    return HarmonicPlanIds(
        harmonic_function_id=HARMONIC_PLAN_UNKNOWN_ID,
        root_degree_id=HARMONIC_PLAN_UNKNOWN_ID,
        root_accidental_id=HARMONIC_PLAN_UNKNOWN_ID,
        quality_id=HARMONIC_PLAN_UNKNOWN_ID,
        extension_id=HARMONIC_PLAN_UNKNOWN_ID,
        chord_change_id=HARMONIC_PLAN_UNKNOWN_ID,
    )


def harmonic_plan_ids_from_window(
    window: HarmonicPlanWindow,
    *,
    chord_changed: bool,
) -> HarmonicPlanIds:
    return harmonic_plan_ids_from_chord(window.chord, chord_changed=chord_changed)


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
    )


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
