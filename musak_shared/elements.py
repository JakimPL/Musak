from enum import StrEnum
from fractions import Fraction
from typing import Final

from musak_shared.misc import is_power_of_two

MUSICXML_EXTENSIONS: Final[frozenset[str]] = frozenset({".xml", ".mxl", ".musicxml"})

MIDI_MAX_PITCH: Final[int] = 127
MIDDLE_C: Final[int] = 60

MAX_NOTES_PER_HAND: Final[int] = 5
MAX_ONSET_SPAN_SEMITONES: Final[int] = 12

PITCHES_PER_OCTAVE: Final[int] = 12
MIDI_OCTAVE_OFFSET: Final[int] = 1
TRIADIC_CONSONANT_INTERVAL_CLASSES: Final[frozenset[int]] = frozenset({0, 3, 4, 5, 7, 8, 9})
PERFECT_CONSONANT_INTERVAL_CLASSES: Final[frozenset[int]] = frozenset({0, 5, 7})
SHARP_PITCH_CLASS_NAMES: Final[tuple[str, ...]] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_PITCH_CLASS_NAMES: Final[tuple[str, ...]] = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
PITCH_CLASS_NAMES: Final[tuple[str, ...]] = SHARP_PITCH_CLASS_NAMES
KEYS: Final[dict[int, str]] = dict(enumerate(PITCH_CLASS_NAMES))
KEY_FIFTHS_MIN: Final[int] = -7
KEY_FIFTHS_MAX: Final[int] = 7


class HarmonicFunction(StrEnum):
    TONIC = "tonic"
    PREDOMINANT = "predominant"
    DOMINANT = "dominant"


HARMONIC_FUNCTION_BY_DEGREE: Final[dict[int, HarmonicFunction]] = {
    1: HarmonicFunction.TONIC,
    2: HarmonicFunction.PREDOMINANT,
    3: HarmonicFunction.TONIC,
    4: HarmonicFunction.PREDOMINANT,
    5: HarmonicFunction.DOMINANT,
    6: HarmonicFunction.TONIC,
    7: HarmonicFunction.DOMINANT,
}

QUARTER_NOTE_DURATION: Final[Fraction] = Fraction(1, 4)
QUARTERS_PER_WHOLE: Final[int] = 4
DEFAULT_TICKS_PER_BEAT: Final[int] = 480
PIANO_PROGRAM: Final[int] = 0
DEFAULT_VELOCITY: Final[int] = 72

DOTTED_LIKE_DURATIONS: Final[frozenset[Fraction]] = frozenset(
    {
        Fraction(3, 4),
        Fraction(3, 8),
        Fraction(3, 16),
        Fraction(3, 32),
        Fraction(3, 64),
    }
)

_MAX_DOT_COUNT: Final[int] = 3


def is_dotted_duration(duration: Fraction) -> bool:
    for dot_count in range(1, _MAX_DOT_COUNT + 1):
        base_duration = duration * Fraction(2**dot_count, 2 ** (dot_count + 1) - 1)
        if base_duration.numerator == 1 and is_power_of_two(base_duration.denominator):
            return True

    return False


def pitch_class_from_key_fifths(key_fifths: int) -> int:
    return (key_fifths * 7) % PITCHES_PER_OCTAVE


def key_fifths_from_pitch_class(pitch_class: int) -> int:
    if not 0 <= pitch_class < PITCHES_PER_OCTAVE:
        raise ValueError(f"pitch class must be in [0, {PITCHES_PER_OCTAVE})")

    candidates = [
        key_fifths
        for key_fifths in range(KEY_FIFTHS_MIN, KEY_FIFTHS_MAX + 1)
        if pitch_class_from_key_fifths(key_fifths) == pitch_class
    ]
    if candidates:
        return min(candidates, key=lambda key_fifths: (abs(key_fifths), key_fifths))

    raise ValueError(f"cannot derive key fifths for pitch class {pitch_class}")
