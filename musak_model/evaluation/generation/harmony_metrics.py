from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from musak_model.conditioning.harmony.extraction import (
    harmonic_plan_windows_from_segment,
)
from musak_model.conditioning.harmony.schema import HarmonicPlanWindow
from musak_model.data.schema import Segment
from musak_model.evaluation.generation.protocols import GenerationEvaluationOptions
from musak_model.evaluation.generation.sampling import segment_from_tokens
from musak_model.evaluation.generation.schema import GenerationSample
from musak_model.harmony.decoding import ChordDecoderConfig, ViterbiChordDecoder
from musak_model.harmony.expansion import chord_pitch_class_set
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.config import RhythmAnalysisConfig
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import degree_pitch_class, note_token_to_midi_pitch
from musak_model.tokens.schema import Hand, ScaleType
from musak_shared.elements import (
    PERFECT_CONSONANT_INTERVAL_CLASSES,
    PITCHES_PER_OCTAVE,
    TRIADIC_CONSONANT_INTERVAL_CLASSES,
    HarmonicFunction,
)

_METRIC_GROUP_NAME: Final[str] = "harmony"


@dataclass(frozen=True)
class _PitchedNoteEvent:
    hand: Hand
    start: Fraction
    end: Fraction
    pitch_class: int
    midi_pitch: int


@dataclass(frozen=True)
class _AgreementCounts:
    overlap_duration: Fraction
    function_match_duration: Fraction
    root_degree_match_duration: Fraction


@dataclass(frozen=True)
class _CoverageCounts:
    note_duration: Fraction
    chord_tone_duration: Fraction
    strong_beat_notes: int
    strong_beat_chord_tones: int


@dataclass(frozen=True)
class _ConsonanceCounts:
    triadic_consonant_pairs: int
    perfect_consonant_pairs: int
    total_pairs: int


def harmonic_plan_metrics(
    suite_name: str,
    samples: list[GenerationSample],
    *,
    config: GenerationEvaluationOptions,
    duration_vocabulary: DurationVocabulary,
    rhythm_config: RhythmAnalysisConfig,
) -> dict[str, float]:
    prefix = f"generation/{suite_name}/{_METRIC_GROUP_NAME}"
    planned_samples = [sample for sample in samples if sample.harmonic_plan_windows is not None]
    decoded_samples = [sample for sample in planned_samples if sample.decode_error is None]
    metrics = {
        f"{prefix}/count/planned_samples": float(len(planned_samples)),
        f"{prefix}/count/decoded_samples": float(len(decoded_samples)),
        f"{prefix}/count/planned_windows": float(
            sum(len(sample.harmonic_plan_windows or ()) for sample in planned_samples)
        ),
    }
    if not decoded_samples:
        return metrics

    chord_vocabulary = ChordVocabularyConfig.load()
    decoder = ViterbiChordDecoder(config=ChordDecoderConfig.load())
    agreement_counts = _AgreementCounts(Fraction(0), Fraction(0), Fraction(0))
    coverage_counts = _CoverageCounts(Fraction(0), Fraction(0), 0, 0)
    consonance_counts = _ConsonanceCounts(0, 0, 0)
    decoded_window_count = 0
    final_slot_samples = 0
    final_slot_closures = 0

    for sample in decoded_samples:
        planned_windows = _required_plan_windows(sample)
        segment = segment_from_tokens(sample.tokens, config=config)
        decoded_windows = harmonic_plan_windows_from_segment(
            segment,
            decoder=decoder,
            duration_vocabulary=duration_vocabulary,
            vocabulary=chord_vocabulary,
        )
        decoded_window_count += len(decoded_windows)
        agreement_counts = _add_agreement_counts(
            agreement_counts,
            _agreement_counts(planned_windows=planned_windows, decoded_windows=decoded_windows),
        )

        note_events = _note_events(segment, duration_vocabulary=duration_vocabulary)
        coverage_counts = _add_coverage_counts(
            coverage_counts,
            _coverage_counts(
                note_events,
                planned_windows=planned_windows,
                scale_type=segment.scale_type,
                chord_vocabulary=chord_vocabulary,
                measure_duration=Fraction(segment.time_numerator, segment.time_denominator),
                strong_beat_offsets=rhythm_config.strong_beat_offsets,
            ),
        )
        consonance_counts = _add_consonance_counts(consonance_counts, _consonance_counts(note_events))

        closure = _final_slot_closure(
            sample,
            note_events=note_events,
            planned_windows=planned_windows,
            scale_type=segment.scale_type,
            chord_vocabulary=chord_vocabulary,
        )
        if closure is not None:
            final_slot_samples += 1
            final_slot_closures += int(closure)

    metrics[f"{prefix}/count/decoded_windows"] = float(decoded_window_count)
    metrics.update(_agreement_metrics(prefix, agreement_counts))
    metrics.update(_coverage_metrics(prefix, coverage_counts))
    metrics.update(_consonance_metrics(prefix, consonance_counts))
    if final_slot_samples > 0:
        metrics[f"{prefix}/rate/final_slot_closure"] = final_slot_closures / final_slot_samples

    return metrics


def _required_plan_windows(sample: GenerationSample) -> tuple[HarmonicPlanWindow, ...]:
    if sample.harmonic_plan_windows is None:
        raise ValueError("harmonic plan metrics require planned samples")

    return sample.harmonic_plan_windows


def _agreement_counts(
    *,
    planned_windows: tuple[HarmonicPlanWindow, ...],
    decoded_windows: tuple[HarmonicPlanWindow, ...],
) -> _AgreementCounts:
    overlap_duration = Fraction(0)
    function_match_duration = Fraction(0)
    root_degree_match_duration = Fraction(0)
    for planned_window, decoded_window, overlap in _overlapping_windows(planned_windows, decoded_windows):
        overlap_duration += overlap
        if planned_window.harmonic_function == decoded_window.harmonic_function:
            function_match_duration += overlap
        if planned_window.chord.root_degree == decoded_window.chord.root_degree:
            root_degree_match_duration += overlap

    return _AgreementCounts(
        overlap_duration=overlap_duration,
        function_match_duration=function_match_duration,
        root_degree_match_duration=root_degree_match_duration,
    )


def _overlapping_windows(
    left_windows: tuple[HarmonicPlanWindow, ...],
    right_windows: tuple[HarmonicPlanWindow, ...],
) -> tuple[tuple[HarmonicPlanWindow, HarmonicPlanWindow, Fraction], ...]:
    overlaps: list[tuple[HarmonicPlanWindow, HarmonicPlanWindow, Fraction]] = []
    left_index = 0
    right_index = 0
    while left_index < len(left_windows) and right_index < len(right_windows):
        left_window = left_windows[left_index]
        right_window = right_windows[right_index]
        start = max(left_window.start, right_window.start)
        end = min(left_window.end, right_window.end)
        if end > start:
            overlaps.append((left_window, right_window, end - start))

        if left_window.end <= right_window.end:
            left_index += 1
        else:
            right_index += 1

    return tuple(overlaps)


def _note_events(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> tuple[_PitchedNoteEvent, ...]:
    runs_by_hand = extract_hand_onset_runs(
        segment.tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=segment.time_numerator,
        time_denominator=segment.time_denominator,
    )
    events: list[_PitchedNoteEvent] = []
    for hand, runs in runs_by_hand.items():
        for run in runs:
            for onset in run.onsets:
                for note in onset.notes:
                    events.append(
                        _PitchedNoteEvent(
                            hand=hand,
                            start=onset.start,
                            end=onset.start + onset.duration,
                            pitch_class=degree_pitch_class(
                                note.degree,
                                note.accidental,
                                scale_type=segment.scale_type,
                            ),
                            midi_pitch=note_token_to_midi_pitch(
                                note,
                                scale_root=segment.scale_root,
                                scale_type=segment.scale_type,
                                hand=hand,
                            ),
                        )
                    )

    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.start,
                event.hand.value,
                event.midi_pitch,
            ),
        )
    )


def _coverage_counts(
    note_events: tuple[_PitchedNoteEvent, ...],
    *,
    planned_windows: tuple[HarmonicPlanWindow, ...],
    scale_type: ScaleType,
    chord_vocabulary: ChordVocabularyConfig,
    measure_duration: Fraction,
    strong_beat_offsets: tuple[Fraction, ...],
) -> _CoverageCounts:
    note_duration = Fraction(0)
    chord_tone_duration = Fraction(0)
    for event in note_events:
        for planned_window, overlap in _event_window_overlaps(event, planned_windows):
            note_duration += overlap
            if _event_matches_chord(
                event,
                planned_window,
                scale_type=scale_type,
                chord_vocabulary=chord_vocabulary,
            ):
                chord_tone_duration += overlap

    strong_beat_notes = 0
    strong_beat_chord_tones = 0
    for event in note_events:
        if not _is_strong_beat(event.start, measure_duration=measure_duration, strong_beat_offsets=strong_beat_offsets):
            continue

        active_window = _window_at_position(planned_windows, event.start)
        if active_window is None:
            continue

        strong_beat_notes += 1
        strong_beat_chord_tones += int(
            _event_matches_chord(
                event,
                active_window,
                scale_type=scale_type,
                chord_vocabulary=chord_vocabulary,
            )
        )

    return _CoverageCounts(
        note_duration=note_duration,
        chord_tone_duration=chord_tone_duration,
        strong_beat_notes=strong_beat_notes,
        strong_beat_chord_tones=strong_beat_chord_tones,
    )


def _event_window_overlaps(
    event: _PitchedNoteEvent,
    windows: tuple[HarmonicPlanWindow, ...],
) -> tuple[tuple[HarmonicPlanWindow, Fraction], ...]:
    overlaps: list[tuple[HarmonicPlanWindow, Fraction]] = []
    for window in windows:
        start = max(event.start, window.start)
        end = min(event.end, window.end)
        if end > start:
            overlaps.append((window, end - start))

    return tuple(overlaps)


def _event_matches_chord(
    event: _PitchedNoteEvent,
    window: HarmonicPlanWindow,
    *,
    scale_type: ScaleType,
    chord_vocabulary: ChordVocabularyConfig,
) -> bool:
    chord_pitch_classes = chord_pitch_class_set(window.chord, scale_type=scale_type, vocabulary=chord_vocabulary)
    return event.pitch_class in chord_pitch_classes


def _is_strong_beat(
    position: Fraction,
    *,
    measure_duration: Fraction,
    strong_beat_offsets: tuple[Fraction, ...],
) -> bool:
    return position % measure_duration in frozenset(strong_beat_offsets)


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


def _consonance_counts(note_events: tuple[_PitchedNoteEvent, ...]) -> _ConsonanceCounts:
    right_events_by_start = _events_by_start(note_events, hand=Hand.RIGHT)
    left_events_by_start = _events_by_start(note_events, hand=Hand.LEFT)
    triadic_consonant_pairs = 0
    perfect_consonant_pairs = 0
    total_pairs = 0
    for start, right_events in right_events_by_start.items():
        left_events = left_events_by_start.get(start, ())
        for right_event in right_events:
            for left_event in left_events:
                total_pairs += 1
                interval_class = abs(right_event.midi_pitch - left_event.midi_pitch) % PITCHES_PER_OCTAVE
                if interval_class in TRIADIC_CONSONANT_INTERVAL_CLASSES:
                    triadic_consonant_pairs += 1
                if interval_class in PERFECT_CONSONANT_INTERVAL_CLASSES:
                    perfect_consonant_pairs += 1

    return _ConsonanceCounts(
        triadic_consonant_pairs=triadic_consonant_pairs,
        perfect_consonant_pairs=perfect_consonant_pairs,
        total_pairs=total_pairs,
    )


def _events_by_start(
    events: tuple[_PitchedNoteEvent, ...],
    *,
    hand: Hand,
) -> dict[Fraction, tuple[_PitchedNoteEvent, ...]]:
    starts = sorted({event.start for event in events if event.hand == hand})
    return {start: tuple(event for event in events if event.hand == hand and event.start == start) for start in starts}


def _final_slot_closure(
    sample: GenerationSample,
    *,
    note_events: tuple[_PitchedNoteEvent, ...],
    planned_windows: tuple[HarmonicPlanWindow, ...],
    scale_type: ScaleType,
    chord_vocabulary: ChordVocabularyConfig,
) -> bool | None:
    if not sample.reached_end or not note_events or not planned_windows:
        return None

    final_window = planned_windows[-1]
    if final_window.harmonic_function != HarmonicFunction.TONIC:
        return False

    final_start = max(event.start for event in note_events)
    final_events = tuple(event for event in note_events if event.start == final_start)
    return any(
        _event_matches_chord(
            event,
            final_window,
            scale_type=scale_type,
            chord_vocabulary=chord_vocabulary,
        )
        for event in final_events
    )


def _agreement_metrics(prefix: str, counts: _AgreementCounts) -> dict[str, float]:
    if counts.overlap_duration == 0:
        return {}

    return {
        f"{prefix}/rate/harmonic_function_agreement": float(counts.function_match_duration / counts.overlap_duration),
        f"{prefix}/rate/root_degree_agreement": float(counts.root_degree_match_duration / counts.overlap_duration),
    }


def _coverage_metrics(prefix: str, counts: _CoverageCounts) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if counts.note_duration > 0:
        metrics[f"{prefix}/rate/duration_weighted_chord_tone_coverage"] = float(
            counts.chord_tone_duration / counts.note_duration
        )

    if counts.strong_beat_notes > 0:
        metrics[f"{prefix}/rate/strong_beat_chord_tone_coverage"] = (
            counts.strong_beat_chord_tones / counts.strong_beat_notes
        )

    return metrics


def _consonance_metrics(prefix: str, counts: _ConsonanceCounts) -> dict[str, float]:
    metrics = {f"{prefix}/count/coincident_onset_pairs": float(counts.total_pairs)}
    if counts.total_pairs > 0:
        metrics[f"{prefix}/rate/coincident_onset_triadic_consonance"] = (
            counts.triadic_consonant_pairs / counts.total_pairs
        )
        metrics[f"{prefix}/rate/coincident_onset_perfect_consonance"] = (
            counts.perfect_consonant_pairs / counts.total_pairs
        )

    return metrics


def _add_agreement_counts(left: _AgreementCounts, right: _AgreementCounts) -> _AgreementCounts:
    return _AgreementCounts(
        overlap_duration=left.overlap_duration + right.overlap_duration,
        function_match_duration=left.function_match_duration + right.function_match_duration,
        root_degree_match_duration=left.root_degree_match_duration + right.root_degree_match_duration,
    )


def _add_coverage_counts(left: _CoverageCounts, right: _CoverageCounts) -> _CoverageCounts:
    return _CoverageCounts(
        note_duration=left.note_duration + right.note_duration,
        chord_tone_duration=left.chord_tone_duration + right.chord_tone_duration,
        strong_beat_notes=left.strong_beat_notes + right.strong_beat_notes,
        strong_beat_chord_tones=left.strong_beat_chord_tones + right.strong_beat_chord_tones,
    )


def _add_consonance_counts(left: _ConsonanceCounts, right: _ConsonanceCounts) -> _ConsonanceCounts:
    return _ConsonanceCounts(
        triadic_consonant_pairs=left.triadic_consonant_pairs + right.triadic_consonant_pairs,
        perfect_consonant_pairs=left.perfect_consonant_pairs + right.perfect_consonant_pairs,
        total_pairs=left.total_pairs + right.total_pairs,
    )
