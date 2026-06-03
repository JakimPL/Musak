from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Final

from musak_model.decoder import PianoRollEvent, segment_to_piano_roll_events
from musak_model.harmony.expansion import chord_pitch_class_set
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.render.renderer import RenderedChord
from musak_model.synthetic.validation.generation import GeneratedSample
from musak_model.synthetic.validation.options import MetricOptions
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand
from musak_shared.elements import PITCHES_PER_OCTAVE

_METRIC_PREFIX: Final[str] = "generation/synthetic"
_STEP_SEMITONES: Final[int] = 2
_LEAP_SEMITONES: Final[int] = 4
_HANDS: Final[tuple[Hand, ...]] = (Hand.RIGHT, Hand.LEFT)


class _Tally:
    def __init__(self) -> None:
        self.strong_hits = 0
        self.strong_total = 0
        self.weak_hits = 0
        self.weak_total = 0
        self.intervals: list[int] = []
        self.chord_onsets = 0
        self.onsets = 0


def synthetic_metrics(
    samples: Sequence[GeneratedSample],
    *,
    options: MetricOptions,
    chord_vocabulary: ChordVocabularyConfig,
    duration_vocabulary: DurationVocabulary,
) -> dict[str, float]:
    bar_duration = Fraction(options.time_numerator, options.time_denominator)
    beat_duration = Fraction(1, options.time_denominator)
    tally = _Tally()
    for sample in samples:
        segment = sample.segment
        if segment is None:
            continue

        try:
            events = segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary)
        except ValueError:
            continue

        spans = _chord_spans(sample.chords, options=options, chord_vocabulary=chord_vocabulary)
        _tally_chord_tones(tally, events, spans=spans, bar_duration=bar_duration, beat_duration=beat_duration)
        _tally_voices(tally, events)

    strong = _rate(tally.strong_hits, tally.strong_total)
    weak = _rate(tally.weak_hits, tally.weak_total)
    metrics = {
        f"{_METRIC_PREFIX}/strong_beat_chord_tone_fraction": strong,
        f"{_METRIC_PREFIX}/weak_beat_chord_tone_fraction": weak,
        f"{_METRIC_PREFIX}/chord_tone_strong_weak_gap": strong - weak,
        f"{_METRIC_PREFIX}/mean_abs_melodic_interval": _mean(tally.intervals),
        f"{_METRIC_PREFIX}/stepwise_fraction": _fraction(tally.intervals, lambda interval: interval <= _STEP_SEMITONES),
        f"{_METRIC_PREFIX}/leap_fraction": _fraction(tally.intervals, lambda interval: interval > _LEAP_SEMITONES),
        f"{_METRIC_PREFIX}/chord_onset_fraction": _rate(tally.chord_onsets, tally.onsets),
    }
    return {name: value for name, value in metrics.items() if math.isfinite(value)}


def _tally_chord_tones(
    tally: _Tally,
    events: Sequence[PianoRollEvent],
    *,
    spans: list[tuple[Fraction, Fraction, frozenset[int]]],
    bar_duration: Fraction,
    beat_duration: Fraction,
) -> None:
    for event in events:
        pitch_classes = _active_pitch_classes(spans, event.start)
        if pitch_classes is None:
            continue

        is_chord_tone = event.midi_pitch % PITCHES_PER_OCTAVE in pitch_classes
        if _on_beat(event.start, bar_duration=bar_duration, beat_duration=beat_duration):
            tally.strong_total += 1
            tally.strong_hits += int(is_chord_tone)
        else:
            tally.weak_total += 1
            tally.weak_hits += int(is_chord_tone)


def _tally_voices(tally: _Tally, events: Sequence[PianoRollEvent]) -> None:
    by_hand: dict[Hand, dict[Fraction, list[int]]] = {hand: defaultdict(list) for hand in _HANDS}
    for event in events:
        by_hand[event.hand][event.start].append(event.midi_pitch)

    for starts in by_hand.values():
        ordered = sorted(starts)
        top_voice = [max(starts[start]) for start in ordered]
        tally.intervals.extend(abs(later - earlier) for earlier, later in zip(top_voice, top_voice[1:]))
        for start in ordered:
            tally.onsets += 1
            if len(starts[start]) >= 2:
                tally.chord_onsets += 1


def _chord_spans(
    chords: Sequence[RenderedChord],
    *,
    options: MetricOptions,
    chord_vocabulary: ChordVocabularyConfig,
) -> list[tuple[Fraction, Fraction, frozenset[int]]]:
    return [
        (
            chord.offset,
            chord.offset + chord.duration,
            frozenset(
                (options.scale_root + interval_class) % PITCHES_PER_OCTAVE
                for interval_class in chord_pitch_class_set(
                    chord.chord, scale_type=options.scale_type, vocabulary=chord_vocabulary
                )
            ),
        )
        for chord in chords
    ]


def _active_pitch_classes(
    spans: list[tuple[Fraction, Fraction, frozenset[int]]],
    start: Fraction,
) -> frozenset[int] | None:
    for offset, end, pitch_classes in spans:
        if offset <= start < end:
            return pitch_classes

    return None


def _on_beat(start: Fraction, *, bar_duration: Fraction, beat_duration: Fraction) -> bool:
    offset_in_bar = start - (start // bar_duration) * bar_duration
    return offset_in_bar % beat_duration == 0


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return math.nan

    return numerator / denominator


def _mean(values: list[int]) -> float:
    if not values:
        return math.nan

    return sum(values) / len(values)


def _fraction(values: list[int], predicate: Callable[[int], bool]) -> float:
    if not values:
        return math.nan

    return sum(bool(predicate(value)) for value in values) / len(values)
