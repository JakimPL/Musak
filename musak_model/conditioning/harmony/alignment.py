from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Final

from musak_model.conditioning.harmony.schema import (
    HarmonicPlanIds,
    HarmonicPlanInputTensors,
    HarmonicPlanWindow,
)
from musak_model.conditioning.harmony.vocabulary import (
    harmonic_plan_ids_from_windows,
    harmonic_plan_tensors_from_ids,
    unknown_harmonic_plan_ids,
)
from musak_model.generation.constraints import GenerationConstraints
from musak_model.generation.coordinates import (
    DecoderInputCoordinates,
    decoder_input_coordinates_from_token_ids,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary

_PADDING_COORDINATE_LIMIT: Final[int] = 0


def harmonic_plan_tensors_from_token_ids(
    prefix_token_ids: Sequence[int],
    *,
    windows: Sequence[HarmonicPlanWindow],
    constraints: GenerationConstraints,
    token_vocabulary: TokenVocabulary,
    duration_vocabulary: DurationVocabulary,
    duration_tick_denominator: int,
) -> HarmonicPlanInputTensors:
    coordinates = decoder_input_coordinates_from_token_ids(
        prefix_token_ids,
        constraints=constraints,
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        duration_tick_denominator=duration_tick_denominator,
    )
    return harmonic_plan_tensors_from_decoder_coordinates(
        windows,
        constraints=constraints,
        coordinates=coordinates,
        duration_tick_denominator=duration_tick_denominator,
    )


def harmonic_plan_tensors_from_decoder_coordinates(
    windows: Sequence[HarmonicPlanWindow],
    *,
    constraints: GenerationConstraints,
    coordinates: DecoderInputCoordinates,
    duration_tick_denominator: int,
) -> HarmonicPlanInputTensors:
    return harmonic_plan_tensors_from_ids(
        harmonic_plan_ids_from_decoder_coordinates(
            windows,
            constraints=constraints,
            coordinates=coordinates,
            duration_tick_denominator=duration_tick_denominator,
        )
    )


def harmonic_plan_ids_from_decoder_coordinates(
    windows: Sequence[HarmonicPlanWindow],
    *,
    constraints: GenerationConstraints,
    coordinates: DecoderInputCoordinates,
    duration_tick_denominator: int,
) -> tuple[HarmonicPlanIds, ...]:
    _validate_coordinate_shapes(coordinates)
    _validate_duration_tick_denominator(duration_tick_denominator)
    _validate_windows(windows)

    window_ids = harmonic_plan_ids_from_windows(windows)
    aligned_ids: list[HarmonicPlanIds] = []
    for bar_index, bar_relative_tick in zip(
        coordinates.bar_indices,
        coordinates.bar_relative_ticks,
        strict=True,
    ):
        if _is_padding_coordinate(bar_index, bar_relative_tick):
            aligned_ids.append(unknown_harmonic_plan_ids())
            continue

        score_position = constraints.bar_start(bar_index) + Fraction(bar_relative_tick, duration_tick_denominator)
        window_index = _window_index_at_position(windows, score_position)
        aligned_ids.append(unknown_harmonic_plan_ids() if window_index is None else window_ids[window_index])

    return tuple(aligned_ids)


def _validate_coordinate_shapes(coordinates: DecoderInputCoordinates) -> None:
    if not (
        len(coordinates.bar_indices)
        == len(coordinates.bar_relative_ticks)
        == len(coordinates.bar_duration_ticks)
        == len(coordinates.active_hand_ids)
    ):
        raise ValueError("decoder coordinate fields must have matching lengths")


def _validate_duration_tick_denominator(duration_tick_denominator: int) -> None:
    if duration_tick_denominator <= 0:
        raise ValueError("duration_tick_denominator must be positive")


def _validate_windows(windows: Sequence[HarmonicPlanWindow]) -> None:
    previous_end: Fraction | None = None
    for window in windows:
        if previous_end is not None and window.start < previous_end:
            raise ValueError("harmonic plan windows must be sorted and non-overlapping")

        previous_end = window.end


def _is_padding_coordinate(bar_index: int, bar_relative_tick: int) -> bool:
    return bar_index < _PADDING_COORDINATE_LIMIT or bar_relative_tick < _PADDING_COORDINATE_LIMIT


def _window_index_at_position(windows: Sequence[HarmonicPlanWindow], position: Fraction) -> int | None:
    for index, window in enumerate(windows):
        if window.start <= position < window.end:
            return index

    if windows and position == windows[-1].end:
        return len(windows) - 1

    return None
