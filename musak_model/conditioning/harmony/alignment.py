from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
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
    remaining_bar_count_to_id,
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
    strict_in_span: bool = False,
) -> HarmonicPlanInputTensors:
    return harmonic_plan_tensors_from_ids(
        harmonic_plan_ids_from_decoder_coordinates(
            windows,
            constraints=constraints,
            coordinates=coordinates,
            duration_tick_denominator=duration_tick_denominator,
            strict_in_span=strict_in_span,
        )
    )


def harmonic_plan_ids_from_decoder_coordinates(
    windows: Sequence[HarmonicPlanWindow],
    *,
    constraints: GenerationConstraints,
    coordinates: DecoderInputCoordinates,
    duration_tick_denominator: int,
    strict_in_span: bool = False,
) -> tuple[HarmonicPlanIds, ...]:
    _validate_coordinate_shapes(coordinates)
    _validate_duration_tick_denominator(duration_tick_denominator)
    _validate_windows(windows)

    window_ids = harmonic_plan_ids_from_windows(windows)
    aligned_windows = harmonic_plan_windows_from_decoder_coordinates(
        windows,
        constraints=constraints,
        coordinates=coordinates,
        duration_tick_denominator=duration_tick_denominator,
        strict_in_span=strict_in_span,
    )
    aligned_ids: list[HarmonicPlanIds] = []
    for bar_index, window in zip(
        coordinates.bar_indices,
        aligned_windows,
        strict=True,
    ):
        if window is None:
            aligned_ids.append(unknown_harmonic_plan_ids())
            continue

        window_index = windows.index(window)
        aligned_ids.append(
            _ids_for_coordinate(
                window_ids[window_index],
                bar_index=bar_index,
                constraints=constraints,
            )
        )

    return tuple(aligned_ids)


def harmonic_plan_windows_from_decoder_coordinates(
    windows: Sequence[HarmonicPlanWindow],
    *,
    constraints: GenerationConstraints,
    coordinates: DecoderInputCoordinates,
    duration_tick_denominator: int,
    strict_in_span: bool = False,
) -> tuple[HarmonicPlanWindow | None, ...]:
    _validate_coordinate_shapes(coordinates)
    _validate_duration_tick_denominator(duration_tick_denominator)
    _validate_windows(windows)

    aligned_windows: list[HarmonicPlanWindow | None] = []
    for bar_index, bar_relative_tick in zip(
        coordinates.bar_indices,
        coordinates.bar_relative_ticks,
        strict=True,
    ):
        if _is_padding_coordinate(bar_index, bar_relative_tick):
            aligned_windows.append(None)
            continue

        score_position = constraints.bar_start(bar_index) + Fraction(bar_relative_tick, duration_tick_denominator)
        window = _window_at_position(windows, score_position)
        if (
            window is None
            and strict_in_span
            and _is_requested_score_position(
                score_position,
                constraints=constraints,
            )
        ):
            raise ValueError(f"no harmonic plan window covers in-span position {score_position}")

        aligned_windows.append(window)

    return tuple(aligned_windows)


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


def _is_requested_score_position(position: Fraction, *, constraints: GenerationConstraints) -> bool:
    if constraints.bar_count <= 0:
        return False

    return constraints.bar_start(0) <= position <= constraints.bar_end(constraints.bar_count - 1)


def _ids_for_coordinate(
    ids: HarmonicPlanIds,
    *,
    bar_index: int,
    constraints: GenerationConstraints,
) -> HarmonicPlanIds:
    return replace(
        ids,
        remaining_bar_id=remaining_bar_count_to_id(_remaining_bar_count(bar_index, constraints=constraints)),
    )


def _remaining_bar_count(bar_index: int, *, constraints: GenerationConstraints) -> int:
    return max(constraints.bar_count - bar_index - 1, 0)


def _window_at_position(windows: Sequence[HarmonicPlanWindow], position: Fraction) -> HarmonicPlanWindow | None:
    for window in windows:
        if window.start <= position < window.end:
            return window

    if windows and position == windows[-1].end:
        return windows[-1]

    return None
