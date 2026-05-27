import json
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from musak_model.n_grams.figure.parser import HandOnsetRun
from musak_model.n_grams.figure.pitch import note_diatonic_position
from musak_model.n_grams.figure.schema import FigureDegree, FigureNGram

type FigureDurationSignature = tuple[int, int]
type FigureOnsetSignature = tuple[tuple[FigureDegree, ...], FigureDurationSignature]
type FigureSignature = tuple[FigureOnsetSignature, ...]

_JSON_SEPARATORS: Final[tuple[str, str]] = (",", ":")


@dataclass(frozen=True)
class RawFigureOnset:
    degrees: tuple[FigureDegree, ...]
    start: Fraction
    held_duration: Fraction
    gap_to_next: Fraction | None


@dataclass(frozen=True)
class FigureOccurrence:
    figure_length: int
    signature: FigureSignature
    anchor_degree: int
    anchor_accidental: int
    anchor_octave: int
    base_duration: Fraction
    start: Fraction


def iter_figure_occurrences_from_run(
    run: HandOnsetRun,
    *,
    min_n: int,
    max_n: int,
    scale_size: int,
) -> Iterator[FigureOccurrence]:
    if min_n <= 0:
        raise ValueError("min_n must be positive")

    if max_n < min_n:
        raise ValueError("max_n must be greater than or equal to min_n")

    raw_onsets = _raw_figure_onsets(run, scale_size=scale_size)
    for start_index in range(len(raw_onsets)):
        largest_window_length = min(max_n, len(raw_onsets) - start_index)
        if largest_window_length < min_n:
            continue

        for window_length in range(min_n, largest_window_length + 1):
            yield _build_figure_occurrence_from_raw_window(
                raw_onsets,
                start_index=start_index,
                window_length=window_length,
                scale_size=scale_size,
            )


def iter_figure_signatures_from_run(
    run: HandOnsetRun,
    *,
    min_n: int,
    max_n: int,
    scale_size: int,
) -> Iterator[tuple[int, FigureSignature]]:
    for occurrence in iter_figure_occurrences_from_run(run, min_n=min_n, max_n=max_n, scale_size=scale_size):
        yield occurrence.figure_length, occurrence.signature


def build_figure_signature_from_raw_window(
    raw_onsets: tuple[RawFigureOnset, ...],
    *,
    start_index: int,
    window_length: int,
) -> FigureSignature:
    if window_length <= 0:
        raise ValueError("window_length must be positive")

    window = raw_onsets[start_index : start_index + window_length]
    if len(window) != window_length:
        return ()

    anchor_position = min(position for position, _ in window[0].degrees)
    durations = tuple(_window_onset_duration(window, onset_index=index) for index in range(window_length))
    normalized_durations = _normalize_duration_signatures(durations)
    return tuple(
        (
            tuple(sorted((position - anchor_position, accidental) for position, accidental in onset.degrees)),
            duration_signature,
        )
        for onset, duration_signature in zip(window, normalized_durations, strict=True)
    )


def _build_figure_occurrence_from_raw_window(
    raw_onsets: tuple[RawFigureOnset, ...],
    *,
    start_index: int,
    window_length: int,
    scale_size: int,
) -> FigureOccurrence:
    window = raw_onsets[start_index : start_index + window_length]
    signature = build_figure_signature_from_raw_window(
        raw_onsets,
        start_index=start_index,
        window_length=window_length,
    )
    anchor_position, anchor_accidental = window[0].degrees[0]
    durations = tuple(_window_onset_duration(window, onset_index=index) for index in range(window_length))
    return FigureOccurrence(
        figure_length=window_length,
        signature=signature,
        anchor_degree=anchor_position % scale_size + 1,
        anchor_accidental=anchor_accidental,
        anchor_octave=anchor_position // scale_size,
        base_duration=min(durations),
        start=window[0].start,
    )


def figure_signature_to_ngram(signature: FigureSignature) -> FigureNGram:
    return FigureNGram(
        onsets=tuple(
            (
                degrees,
                Fraction(duration_numerator, duration_denominator),
            )
            for degrees, (duration_numerator, duration_denominator) in signature
        )
    )


def figure_signature_to_json(signature: FigureSignature) -> str:
    return json.dumps(signature, separators=_JSON_SEPARATORS)


def figure_signature_from_json(payload: str) -> FigureSignature:
    parsed = json.loads(payload)
    return tuple(
        (
            tuple((int(position), int(accidental)) for position, accidental in degrees),
            (int(duration[0]), int(duration[1])),
        )
        for degrees, duration in parsed
    )


def figure_signature_monophonic(signature: FigureSignature) -> bool:
    return all(len(degrees) == 1 for degrees, _ in signature)


def figure_signature_chords_only(signature: FigureSignature) -> bool:
    return all(len(degrees) > 1 for degrees, _ in signature)


def figure_signature_in_scale(signature: FigureSignature) -> bool:
    return all(accidental == 0 for degrees, _ in signature for _, accidental in degrees)


def _raw_figure_onsets(
    run: HandOnsetRun,
    *,
    scale_size: int,
) -> tuple[RawFigureOnset, ...]:
    raw_onsets: list[RawFigureOnset] = []
    for onset_index, onset in enumerate(run.onsets):
        next_onset = run.onsets[onset_index + 1] if onset_index + 1 < len(run.onsets) else None
        raw_onsets.append(
            RawFigureOnset(
                degrees=tuple(
                    sorted(
                        (note_diatonic_position(note, scale_size=scale_size), note.accidental) for note in onset.notes
                    )
                ),
                start=onset.start,
                held_duration=onset.duration,
                gap_to_next=next_onset.start - onset.start if next_onset is not None else None,
            )
        )

    return tuple(raw_onsets)


def _window_onset_duration(
    window: tuple[RawFigureOnset, ...],
    *,
    onset_index: int,
) -> Fraction:
    if onset_index == len(window) - 1:
        return window[onset_index].held_duration

    gap_to_next = window[onset_index].gap_to_next
    if gap_to_next is None:
        raise ValueError("internal figure onset is missing gap_to_next")

    return gap_to_next


def _normalize_duration_signatures(durations: tuple[Fraction, ...]) -> tuple[FigureDurationSignature, ...]:
    if any(duration <= 0 for duration in durations):
        raise ValueError("durations must be positive")

    shortest_duration = min(durations)
    return tuple(_duration_signature(duration / shortest_duration) for duration in durations)


def _duration_signature(duration: Fraction) -> FigureDurationSignature:
    return duration.numerator, duration.denominator
