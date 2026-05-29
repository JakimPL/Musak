from dataclasses import dataclass
from fractions import Fraction

from musak_model.data.schema import Segment
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.synthetic.harmony.windows import chord_window_grid
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import degree_pitch_class


@dataclass(frozen=True)
class SoundingWindow:
    start: Fraction
    end: Fraction
    pitch_class_weights: dict[int, Fraction]


def sounding_windows(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    resolution: int,
) -> tuple[SoundingWindow, ...]:
    events = _sounding_events(segment, duration_vocabulary=duration_vocabulary)
    measure_duration = Fraction(segment.time_numerator, segment.time_denominator)
    total_duration = measure_duration * segment.bar_count
    if events:
        total_duration = max(total_duration, max(end for _, end, _ in events))

    windows: list[SoundingWindow] = []
    for window_start, window_end in chord_window_grid(
        measure_duration=measure_duration, total_duration=total_duration, resolution=resolution
    ):
        weights: dict[int, Fraction] = {}
        for event_start, event_end, event_pitch_class in events:
            overlap = min(event_end, window_end) - max(event_start, window_start)
            if overlap > 0:
                weights[event_pitch_class] = weights.get(event_pitch_class, Fraction(0)) + overlap

        windows.append(SoundingWindow(start=window_start, end=window_end, pitch_class_weights=weights))

    return tuple(windows)


def _sounding_events(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> tuple[tuple[Fraction, Fraction, int], ...]:
    runs_by_hand = extract_hand_onset_runs(
        segment.tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=segment.time_numerator,
        time_denominator=segment.time_denominator,
    )
    events: list[tuple[Fraction, Fraction, int]] = []
    for runs in runs_by_hand.values():
        for run in runs:
            for onset in run.onsets:
                for note in onset.notes:
                    event_pitch_class = degree_pitch_class(note.degree, note.accidental, scale_type=segment.scale_type)
                    events.append((onset.start, onset.start + onset.duration, event_pitch_class))

    return tuple(events)
