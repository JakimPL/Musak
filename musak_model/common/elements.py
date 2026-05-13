from typing import Final

PITCHES_PER_OCTAVE: Final[int] = 12
MIDI_OCTAVE_OFFSET: Final[int] = 1

VALID_MODES: Final[tuple[str, ...]] = ("major", "minor")

VALID_SCALE_TYPES: Final[tuple[str, ...]] = (
    "major",
    "natural_minor",
    "harmonic_minor",
    "melodic_minor",
    "dorian",
    "phrygian",
    "lydian",
    "mixolydian",
    "locrian",
)

SCALE_TYPE_INTERVALS: Final[dict[str, tuple[int, ...]]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
}

DOTTED_DURATION_VALUES: Final[tuple[str, ...]] = (
    "dotted_half",
    "dotted_quarter",
    "dotted_eighth",
)
