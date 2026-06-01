from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Final, NamedTuple

from musak_model.data.schema import Segment
from musak_model.harmony.decoding.schema import ChordDecoder, ChordWindow
from musak_model.harmony.diatonic import natural_triad
from musak_model.harmony.expansion import chord_pitch_class_set
from musak_model.harmony.schema import Chord
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.figure.counter import count_figure_ngrams
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.processes.accent import indispensability_per_position
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import ScaleType, scale_size_for_type
from musak_shared.elements import HARMONIC_FUNCTION_BY_DEGREE, HarmonicFunction

TONIC_TRIAD_DEGREE: Final[int] = 1


class _ChordSpan(NamedTuple):
    start: Fraction
    end: Fraction
    chord: Chord


@dataclass(frozen=True)
class HarmonicSlot:
    start: Fraction
    end: Fraction
    bar_index: int
    chord: Chord
    function: HarmonicFunction | None
    metrical_weight: float
    tonic_triad_overlap: float
    dwell: float


@dataclass(frozen=True)
class AnalyzedPiece:
    scale_type: ScaleType
    bar_count: int
    bar_duration: Fraction
    slots: tuple[HarmonicSlot, ...]
    bar_figures: tuple[Counter[FigureNGram], ...]


def analyze_segment(
    segment: Segment,
    *,
    chord_decoder: ChordDecoder,
    chord_vocabulary: ChordVocabularyConfig,
    duration_vocabulary: DurationVocabulary,
    figure_min_n: int,
    figure_max_n: int,
) -> AnalyzedPiece | None:
    bar_count = segment.bar_count
    if bar_count <= 0:
        return None

    windows = chord_decoder.decode(segment, duration_vocabulary=duration_vocabulary, vocabulary=chord_vocabulary)
    if not windows:
        return None

    bar_duration = Fraction(segment.time_numerator, segment.time_denominator)
    return AnalyzedPiece(
        scale_type=segment.scale_type,
        bar_count=bar_count,
        bar_duration=bar_duration,
        slots=_harmonic_slots(
            windows, scale_type=segment.scale_type, chord_vocabulary=chord_vocabulary, bar_duration=bar_duration
        ),
        bar_figures=_bar_figures(
            segment,
            duration_vocabulary=duration_vocabulary,
            bar_duration=bar_duration,
            bar_count=bar_count,
            figure_min_n=figure_min_n,
            figure_max_n=figure_max_n,
        ),
    )


def _harmonic_slots(
    windows: tuple[ChordWindow, ...],
    *,
    scale_type: ScaleType,
    chord_vocabulary: ChordVocabularyConfig,
    bar_duration: Fraction,
) -> tuple[HarmonicSlot, ...]:
    window_duration = min(window.end - window.start for window in windows)
    cells_per_bar = max(1, round(bar_duration / window_duration))
    metrical_weights = indispensability_per_position(cells_per_bar)
    tonic_pitch_classes = chord_pitch_class_set(
        natural_triad(scale_type, TONIC_TRIAD_DEGREE), scale_type=scale_type, vocabulary=chord_vocabulary
    )

    merged: list[_ChordSpan] = []
    for window in windows:
        if merged and merged[-1].chord == window.chord and merged[-1].end == window.start:
            merged[-1] = _ChordSpan(start=merged[-1].start, end=window.end, chord=merged[-1].chord)
        else:
            merged.append(_ChordSpan(start=window.start, end=window.end, chord=window.chord))

    slots: list[HarmonicSlot] = []
    for span in merged:
        bar_index = int(span.start / bar_duration)
        cell_index = _cell_index(span.start - bar_index * bar_duration, window_duration, cells_per_bar)
        chord_pitch_classes = chord_pitch_class_set(span.chord, scale_type=scale_type, vocabulary=chord_vocabulary)
        slots.append(
            HarmonicSlot(
                start=span.start,
                end=span.end,
                bar_index=bar_index,
                chord=span.chord,
                function=HARMONIC_FUNCTION_BY_DEGREE.get(span.chord.root_degree),
                metrical_weight=float(metrical_weights[cell_index]),
                tonic_triad_overlap=len(chord_pitch_classes & tonic_pitch_classes) / len(tonic_pitch_classes),
                dwell=float((span.end - span.start) / bar_duration),
            )
        )

    return tuple(slots)


def _cell_index(offset_in_bar: Fraction, window_duration: Fraction, cells_per_bar: int) -> int:
    return round(offset_in_bar / window_duration) % cells_per_bar


def _bar_figures(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    bar_duration: Fraction,
    bar_count: int,
    figure_min_n: int,
    figure_max_n: int,
) -> tuple[Counter[FigureNGram], ...]:
    scale_size = scale_size_for_type(segment.scale_type)
    runs_by_hand = extract_hand_onset_runs(
        segment.tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=segment.time_numerator,
        time_denominator=segment.time_denominator,
    )
    bar_figures: list[Counter[FigureNGram]] = [Counter() for _ in range(bar_count)]
    for runs in runs_by_hand.values():
        for run in runs:
            if not run.onsets:
                continue

            bar_index = int(run.onsets[0].start / bar_duration)
            if not 0 <= bar_index < bar_count:
                continue

            for counter in count_figure_ngrams(
                [run],
                min_n=figure_min_n,
                max_n=figure_max_n,
                scale_size=scale_size,
            ).values():
                bar_figures[bar_index].update(counter)

    return tuple(bar_figures)
