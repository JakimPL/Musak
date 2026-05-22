from fractions import Fraction

from musak_model.data.schema import ParsedBar, ParsedChord, ParsedNote
from musak_shared.elements import PITCHES_PER_OCTAVE


def pitch_class_histogram(bars: list[ParsedBar]) -> dict[int, Fraction]:
    histogram = {pitch_class: Fraction(0) for pitch_class in range(PITCHES_PER_OCTAVE)}
    for bar in bars:
        for event in bar.events:
            match event:
                case ParsedNote():
                    histogram[event.midi_pitch % PITCHES_PER_OCTAVE] += event.duration
                case ParsedChord():
                    for midi_pitch in event.midi_pitches:
                        histogram[midi_pitch % PITCHES_PER_OCTAVE] += event.duration
    return histogram


def normalized_histogram(histogram: dict[int, Fraction]) -> dict[int, Fraction]:
    normalized = {pitch_class: Fraction(0) for pitch_class in range(PITCHES_PER_OCTAVE)}
    for pitch_class, weight in histogram.items():
        if pitch_class < 0 or pitch_class >= PITCHES_PER_OCTAVE:
            raise ValueError(f"pitch class must be in [0, {PITCHES_PER_OCTAVE - 1}], got {pitch_class}")
        if weight < 0:
            raise ValueError("pitch-class weights must be non-negative")

        normalized[pitch_class] += weight

    return normalized


def declared_key_fifths(bars: list[ParsedBar]) -> int | None:
    for bar in bars:
        if bar.declared_key_fifths is not None:
            return bar.declared_key_fifths

    return None
