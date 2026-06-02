from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import pandas as pd

from musak_model.conditioning.harmony.extraction import harmonic_plan_windows_from_segment
from musak_model.conditioning.harmony.schema import HarmonicPlanWindow
from musak_model.data.schema import Segment
from musak_model.decoder import PianoRollEvent, segment_to_piano_roll_events
from musak_model.harmony.decoding import ChordDecoderConfig, ViterbiChordDecoder
from musak_model.harmony.expansion import chord_pitch_class_set
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand
from musak_shared.elements import (
    PERFECT_CONSONANT_INTERVAL_CLASSES,
    PITCH_CLASS_NAMES,
    PITCHES_PER_OCTAVE,
    TRIADIC_CONSONANT_INTERVAL_CLASSES,
)
from musak_shared.ratios import format_ratio
from notebooks.utils.baselines import chord_label
from notebooks.utils.piano_roll import ChordHighlight, PitchSpelling, midi_pitch_name

_WINDOW_COLUMNS: Final[tuple[str, ...]] = (
    "start_bar",
    "end_bar",
    "label",
    "function",
    "quality",
    "extension",
    "chord_pitch_classes",
    "sounding_pitch_classes",
    "non_chord_pitch_classes",
    "chord_tone_coverage",
    "strong_beat_chord_tone_coverage",
    "strong_beat_non_chord_notes",
)
_NOTE_COLUMNS: Final[tuple[str, ...]] = (
    "hand",
    "pitch",
    "midi_pitch",
    "start_bar",
    "end_bar",
    "duration",
    "token",
    "active_chord",
    "harmonic_function",
    "chord_tone",
    "strong_beat",
    "strong_beat_non_chord",
    "coincident_interval_classes",
    "triadic_consonant_with_other_hand",
    "perfect_consonant_with_other_hand",
)
_PERCENTAGE_DECIMALS: Final[int] = 1


@dataclass(frozen=True)
class HarmonicPlanInspection:
    windows: tuple[HarmonicPlanWindow, ...]
    chord_highlights: tuple[ChordHighlight, ...]
    window_frame: pd.DataFrame
    note_frame: pd.DataFrame
    summary_rows: list[dict[str, str]]


@dataclass(frozen=True)
class _CoverageCounts:
    note_duration: Fraction
    chord_tone_duration: Fraction
    strong_beat_notes: int
    strong_beat_chord_tones: int


@dataclass(frozen=True)
class _ConsonanceCounts:
    total_pairs: int
    triadic_pairs: int
    perfect_pairs: int


def harmonic_plan_inspection(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    pitch_spelling: PitchSpelling = PitchSpelling.SHARPS,
) -> HarmonicPlanInspection:
    chord_vocabulary = ChordVocabularyConfig.load()
    decoder = ViterbiChordDecoder(config=ChordDecoderConfig.load())
    windows = harmonic_plan_windows_from_segment(
        segment,
        decoder=decoder,
        duration_vocabulary=duration_vocabulary,
        vocabulary=chord_vocabulary,
    )
    events = tuple(segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary))
    strong_beat_offsets = NGramAnalysisConfig.load().rhythm_analysis.strong_beat_offsets
    note_frame = _note_frame(
        segment,
        events=events,
        windows=windows,
        duration_vocabulary=duration_vocabulary,
        chord_vocabulary=chord_vocabulary,
        strong_beat_offsets=strong_beat_offsets,
        pitch_spelling=pitch_spelling,
    )
    window_frame = _window_frame(
        segment,
        events=events,
        windows=windows,
        chord_vocabulary=chord_vocabulary,
        strong_beat_offsets=strong_beat_offsets,
        pitch_spelling=pitch_spelling,
    )
    return HarmonicPlanInspection(
        windows=windows,
        chord_highlights=harmonic_plan_chord_highlights(
            segment,
            windows=windows,
            vocabulary=chord_vocabulary,
        ),
        window_frame=window_frame,
        note_frame=note_frame,
        summary_rows=_summary_rows(
            window_frame=window_frame,
            note_frame=note_frame,
            consonance_counts=_coincident_pair_counts(events),
        ),
    )


def harmonic_plan_chord_highlights(
    segment: Segment,
    *,
    windows: Sequence[HarmonicPlanWindow],
    vocabulary: ChordVocabularyConfig,
) -> tuple[ChordHighlight, ...]:
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    return tuple(
        ChordHighlight(
            start_in_bars=_display_bar(segment, window.start, measure_duration=measure_duration),
            end_in_bars=_display_bar(segment, window.end, measure_duration=measure_duration),
            pitch_classes=_absolute_chord_pitch_classes(segment, window, vocabulary=vocabulary),
            label=chord_label(window.chord),
        )
        for window in windows
    )


def _note_frame(
    segment: Segment,
    *,
    events: tuple[PianoRollEvent, ...],
    windows: tuple[HarmonicPlanWindow, ...],
    duration_vocabulary: DurationVocabulary,
    chord_vocabulary: ChordVocabularyConfig,
    strong_beat_offsets: tuple[Fraction, ...],
    pitch_spelling: PitchSpelling,
) -> pd.DataFrame:
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    rows: list[dict[str, object]] = []
    events_by_start_and_hand = _events_by_start_and_hand(events)
    for event in events:
        active_window = _window_at_position(windows, event.start)
        active_pitch_classes = (
            frozenset()
            if active_window is None
            else _absolute_chord_pitch_classes(segment, active_window, vocabulary=chord_vocabulary)
        )
        chord_tone = event.midi_pitch % PITCHES_PER_OCTAVE in active_pitch_classes
        strong_beat = _is_strong_beat(
            event.start,
            measure_duration=measure_duration,
            strong_beat_offsets=strong_beat_offsets,
        )
        coincident_intervals = _coincident_interval_classes(event, events_by_start_and_hand)
        rows.append(
            {
                "hand": event.hand.value,
                "pitch": midi_pitch_name(event.midi_pitch, pitch_spelling=pitch_spelling),
                "midi_pitch": event.midi_pitch,
                "start_bar": _display_bar(segment, event.start, measure_duration=measure_duration),
                "end_bar": _display_bar(segment, event.end, measure_duration=measure_duration),
                "duration": format_ratio(event.duration, separator=":"),
                "token": "" if event.token_text is None else event.token_text,
                "active_chord": "" if active_window is None else chord_label(active_window.chord),
                "harmonic_function": _harmonic_function_text(active_window),
                "chord_tone": chord_tone,
                "strong_beat": strong_beat,
                "strong_beat_non_chord": strong_beat and not chord_tone,
                "coincident_interval_classes": _pitch_class_set_text(coincident_intervals),
                "triadic_consonant_with_other_hand": bool(
                    coincident_intervals and coincident_intervals.issubset(TRIADIC_CONSONANT_INTERVAL_CLASSES)
                ),
                "perfect_consonant_with_other_hand": bool(
                    coincident_intervals and coincident_intervals.issubset(PERFECT_CONSONANT_INTERVAL_CLASSES)
                ),
            }
        )

    return pd.DataFrame(rows, columns=list(_NOTE_COLUMNS))


def _window_frame(
    segment: Segment,
    *,
    events: tuple[PianoRollEvent, ...],
    windows: tuple[HarmonicPlanWindow, ...],
    chord_vocabulary: ChordVocabularyConfig,
    strong_beat_offsets: tuple[Fraction, ...],
    pitch_spelling: PitchSpelling,
) -> pd.DataFrame:
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    rows: list[dict[str, object]] = []
    for window in windows:
        chord_pitch_classes = _absolute_chord_pitch_classes(segment, window, vocabulary=chord_vocabulary)
        sounding_pitch_classes = _sounding_pitch_classes(events, window)
        non_chord_pitch_classes = sounding_pitch_classes - chord_pitch_classes
        coverage = _window_coverage(
            events,
            window=window,
            chord_pitch_classes=chord_pitch_classes,
            measure_duration=measure_duration,
            strong_beat_offsets=strong_beat_offsets,
        )
        rows.append(
            {
                "start_bar": _display_bar(segment, window.start, measure_duration=measure_duration),
                "end_bar": _display_bar(segment, window.end, measure_duration=measure_duration),
                "label": chord_label(window.chord),
                "function": _harmonic_function_text(window),
                "quality": window.chord.quality.value,
                "extension": window.chord.extension.value,
                "chord_pitch_classes": _pitch_class_set_text(chord_pitch_classes),
                "sounding_pitch_classes": _pitch_class_set_text(sounding_pitch_classes),
                "non_chord_pitch_classes": _pitch_class_set_text(non_chord_pitch_classes),
                "chord_tone_coverage": _fraction_or_none(
                    coverage.chord_tone_duration,
                    denominator=coverage.note_duration,
                ),
                "strong_beat_chord_tone_coverage": _rate_or_none(
                    coverage.strong_beat_chord_tones,
                    denominator=coverage.strong_beat_notes,
                ),
                "strong_beat_non_chord_notes": coverage.strong_beat_notes - coverage.strong_beat_chord_tones,
            }
        )

    return pd.DataFrame(rows, columns=list(_WINDOW_COLUMNS))


def _summary_rows(
    *,
    window_frame: pd.DataFrame,
    note_frame: pd.DataFrame,
    consonance_counts: _ConsonanceCounts,
) -> list[dict[str, str]]:
    if note_frame.empty:
        return [
            {"Metric": "Decoded chord windows", "Value": str(len(window_frame))},
            {"Metric": "Decoded note events", "Value": "0"},
        ]

    note_count = len(note_frame)
    chord_tone_count = int(note_frame["chord_tone"].sum())
    strong_beat_count = int(note_frame["strong_beat"].sum())
    strong_beat_chord_tones = int((note_frame["strong_beat"] & note_frame["chord_tone"]).sum())
    return [
        {"Metric": "Decoded chord windows", "Value": str(len(window_frame))},
        {"Metric": "Decoded note events", "Value": str(note_count)},
        {"Metric": "Note-event chord tones", "Value": _format_rate(chord_tone_count, denominator=note_count)},
        {
            "Metric": "Strong-beat chord tones",
            "Value": _format_rate(strong_beat_chord_tones, denominator=strong_beat_count),
        },
        {"Metric": "Strong-beat non-chord notes", "Value": str(strong_beat_count - strong_beat_chord_tones)},
        {
            "Metric": "Triadic coincident-pair consonance",
            "Value": _format_rate(consonance_counts.triadic_pairs, denominator=consonance_counts.total_pairs),
        },
        {
            "Metric": "Perfect coincident-pair consonance",
            "Value": _format_rate(consonance_counts.perfect_pairs, denominator=consonance_counts.total_pairs),
        },
    ]


def _window_coverage(
    events: tuple[PianoRollEvent, ...],
    *,
    window: HarmonicPlanWindow,
    chord_pitch_classes: frozenset[int],
    measure_duration: Fraction,
    strong_beat_offsets: tuple[Fraction, ...],
) -> _CoverageCounts:
    note_duration = Fraction(0)
    chord_tone_duration = Fraction(0)
    strong_beat_notes = 0
    strong_beat_chord_tones = 0
    for event in events:
        overlap = min(event.end, window.end) - max(event.start, window.start)
        if overlap <= 0:
            continue

        chord_tone = event.midi_pitch % PITCHES_PER_OCTAVE in chord_pitch_classes
        note_duration += overlap
        if chord_tone:
            chord_tone_duration += overlap

        if window.start <= event.start < window.end and _is_strong_beat(
            event.start, measure_duration=measure_duration, strong_beat_offsets=strong_beat_offsets
        ):
            strong_beat_notes += 1
            strong_beat_chord_tones += int(chord_tone)

    return _CoverageCounts(
        note_duration=note_duration,
        chord_tone_duration=chord_tone_duration,
        strong_beat_notes=strong_beat_notes,
        strong_beat_chord_tones=strong_beat_chord_tones,
    )


def _sounding_pitch_classes(events: tuple[PianoRollEvent, ...], window: HarmonicPlanWindow) -> frozenset[int]:
    return frozenset(
        event.midi_pitch % PITCHES_PER_OCTAVE
        for event in events
        if min(event.end, window.end) - max(event.start, window.start) > 0
    )


def _absolute_chord_pitch_classes(
    segment: Segment,
    window: HarmonicPlanWindow,
    *,
    vocabulary: ChordVocabularyConfig,
) -> frozenset[int]:
    return frozenset(
        (segment.scale_root + pitch_class) % PITCHES_PER_OCTAVE
        for pitch_class in chord_pitch_class_set(window.chord, scale_type=segment.scale_type, vocabulary=vocabulary)
    )


def _events_by_start_and_hand(
    events: tuple[PianoRollEvent, ...],
) -> dict[tuple[Fraction, Hand], tuple[PianoRollEvent, ...]]:
    grouped: dict[tuple[Fraction, Hand], list[PianoRollEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.start, event.hand)].append(event)

    return {key: tuple(value) for key, value in grouped.items()}


def _coincident_pair_counts(events: tuple[PianoRollEvent, ...]) -> _ConsonanceCounts:
    events_by_start_and_hand = _events_by_start_and_hand(events)
    total_pairs = 0
    triadic_pairs = 0
    perfect_pairs = 0
    for (start, hand), hand_events in events_by_start_and_hand.items():
        if hand != Hand.RIGHT:
            continue

        left_events = events_by_start_and_hand.get((start, Hand.LEFT), ())
        for right_event in hand_events:
            for left_event in left_events:
                interval_class = abs(right_event.midi_pitch - left_event.midi_pitch) % PITCHES_PER_OCTAVE
                total_pairs += 1
                triadic_pairs += int(interval_class in TRIADIC_CONSONANT_INTERVAL_CLASSES)
                perfect_pairs += int(interval_class in PERFECT_CONSONANT_INTERVAL_CLASSES)

    return _ConsonanceCounts(total_pairs=total_pairs, triadic_pairs=triadic_pairs, perfect_pairs=perfect_pairs)


def _coincident_interval_classes(
    event: PianoRollEvent,
    events_by_start_and_hand: dict[tuple[Fraction, Hand], tuple[PianoRollEvent, ...]],
) -> frozenset[int]:
    other_hand = Hand.LEFT if event.hand == Hand.RIGHT else Hand.RIGHT
    return frozenset(
        abs(event.midi_pitch - other_event.midi_pitch) % PITCHES_PER_OCTAVE
        for other_event in events_by_start_and_hand.get((event.start, other_hand), ())
    )


def _window_at_position(
    windows: tuple[HarmonicPlanWindow, ...],
    position: Fraction,
) -> HarmonicPlanWindow | None:
    for window in windows:
        if window.start <= position < window.end:
            return window

    if windows and position == windows[-1].end:
        return windows[-1]

    return None


def _is_strong_beat(
    position: Fraction,
    *,
    measure_duration: Fraction,
    strong_beat_offsets: tuple[Fraction, ...],
) -> bool:
    return position % measure_duration in frozenset(strong_beat_offsets)


def _display_bar(segment: Segment, position: Fraction, *, measure_duration: Fraction) -> float:
    return float(segment.metadata.window_start_bar + 1 + position / measure_duration)


def _harmonic_function_text(window: HarmonicPlanWindow | None) -> str:
    if window is None or window.harmonic_function is None:
        return ""

    return window.harmonic_function.value


def _pitch_class_set_text(pitch_classes: frozenset[int]) -> str:
    return " ".join(PITCH_CLASS_NAMES[pitch_class] for pitch_class in sorted(pitch_classes))


def _fraction_or_none(numerator: Fraction, *, denominator: Fraction) -> float | None:
    if denominator == 0:
        return None

    return float(numerator / denominator)


def _rate_or_none(numerator: int, *, denominator: int) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def _format_rate(numerator: int, *, denominator: int) -> str:
    if denominator == 0:
        return "n/a"

    percentage = round(numerator / denominator * 100, _PERCENTAGE_DECIMALS)
    return f"{percentage}% ({numerator}/{denominator})"
