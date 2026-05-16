from fractions import Fraction
from typing import Final

from musak_shared.common import is_power_of_two

MIDI_MAX_PITCH: Final[int] = 127

PITCHES_PER_OCTAVE: Final[int] = 12
MIDI_OCTAVE_OFFSET: Final[int] = 1
SHARP_PITCH_CLASS_NAMES: Final[tuple[str, ...]] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_PITCH_CLASS_NAMES: Final[tuple[str, ...]] = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
PITCH_CLASS_NAMES: Final[tuple[str, ...]] = SHARP_PITCH_CLASS_NAMES
KEYS: Final[dict[int, str]] = dict(enumerate(PITCH_CLASS_NAMES))

QUARTER_NOTE_DURATION: Final[Fraction] = Fraction(1, 4)

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
