from collections.abc import Sequence
from fractions import Fraction

from musak_model.n_grams.figure.parser import HandOnsetRun, PitchedOnset
from musak_model.n_grams.figure.schema import FigureDegree, FigureNGram, FigureOnset
from musak_model.n_grams.figure.signature import (
    figure_signature_to_ngram,
    iter_figure_signatures_from_run,
)
from musak_model.tokens.pitch import note_diatonic_position
from musak_model.tokens.schema import NoteToken


def build_figure_ngram(
    note_onsets: Sequence[Sequence[NoteToken]],
    durations: Sequence[Fraction],
    *,
    scale_size: int,
) -> FigureNGram:
    if len(note_onsets) != len(durations):
        raise ValueError("note_onsets and durations must have the same length")

    if not note_onsets:
        raise ValueError("note_onsets must not be empty")

    if any(not onset for onset in note_onsets):
        raise ValueError("every onset must contain at least one note")

    normalized_durations = _normalize_durations(durations)
    anchor_position = min(note_diatonic_position(note, scale_size=scale_size) for note in note_onsets[0])
    onsets: list[FigureOnset] = []
    for notes, normalized_duration in zip(note_onsets, normalized_durations, strict=True):
        pitches = tuple(
            sorted(
                _relative_pitch(
                    note,
                    anchor_position=anchor_position,
                    scale_size=scale_size,
                )
                for note in notes
            )
        )
        onsets.append((pitches, normalized_duration))

    return FigureNGram(onsets=tuple(onsets))


def build_figure_ngrams_from_run(
    run: HandOnsetRun,
    *,
    n: int,
    scale_size: int,
) -> tuple[FigureNGram, ...]:
    if n <= 0:
        raise ValueError("n must be positive")

    if len(run.onsets) < n:
        return ()

    return tuple(
        figure_signature_to_ngram(signature)
        for _, signature in iter_figure_signatures_from_run(
            run,
            min_n=n,
            max_n=n,
            scale_size=scale_size,
        )
    )


def build_figure_ngrams_from_runs(
    runs: Sequence[HandOnsetRun],
    *,
    min_n: int,
    max_n: int,
    scale_size: int,
) -> dict[int, tuple[FigureNGram, ...]]:
    if min_n <= 0:
        raise ValueError("min_n must be positive")

    if max_n < min_n:
        raise ValueError("max_n must be greater than or equal to min_n")

    return {
        n: tuple(
            figure
            for run in runs
            for figure in build_figure_ngrams_from_run(
                run,
                n=n,
                scale_size=scale_size,
            )
        )
        for n in range(min_n, max_n + 1)
    }


def _relative_pitch(
    token: NoteToken,
    *,
    anchor_position: int,
    scale_size: int,
) -> FigureDegree:
    position = note_diatonic_position(token, scale_size=scale_size)
    return position - anchor_position, token.accidental


def _build_figure_ngram_from_onsets(
    onsets: tuple[PitchedOnset, ...],
    *,
    scale_size: int,
) -> FigureNGram:
    note_onsets = tuple(onset.notes for onset in onsets)
    durations = tuple(_figure_onset_duration(onsets, onset_index=index) for index in range(len(onsets)))
    return build_figure_ngram(
        note_onsets,
        durations,
        scale_size=scale_size,
    )


def _figure_onset_duration(
    onsets: tuple[PitchedOnset, ...],
    *,
    onset_index: int,
) -> Fraction:
    onset = onsets[onset_index]
    if onset_index == len(onsets) - 1:
        return onset.duration

    return onsets[onset_index + 1].start - onset.start


def _normalize_durations(durations: Sequence[Fraction]) -> tuple[Fraction, ...]:
    if any(duration <= 0 for duration in durations):
        raise ValueError("durations must be positive")

    shortest_duration = min(durations)
    return tuple(duration / shortest_duration for duration in durations)
