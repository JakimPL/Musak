from fractions import Fraction
from typing import Final

MIDI_MAX_PITCH: Final[int] = 127

PITCHES_PER_OCTAVE: Final[int] = 12
MIDI_OCTAVE_OFFSET: Final[int] = 1

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
