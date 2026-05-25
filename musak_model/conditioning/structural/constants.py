from enum import StrEnum
from typing import Final

UNKNOWN_CONTROL_ID: Final[int] = 0
FALSE_CONTROL_ID: Final[int] = 1
TRUE_CONTROL_ID: Final[int] = 2
BOOLEAN_CONTROL_VOCABULARY_SIZE: Final[int] = 3


class StructuralControlName(StrEnum):
    SHORTEST_NOTE_DURATION = "shortest_note_duration"
    HAS_DOTTED_NOTES = "has_dotted_notes"
    MAX_NOTES_PER_ONSET = "max_notes_per_onset"
    MAX_NOTES_PER_HAND = "max_notes_per_hand"
    MAX_ONSET_SPAN_SEMITONES = "max_onset_span_semitones"
    MAX_MELODIC_GAP_SEMITONES = "max_melodic_gap_semitones"
    STATIC_HAND_SPAN_DEGREES = "static_hand_span_degrees"
    BAR_COUNT = "bar_count"


STRUCTURAL_CONTROL_ORDER: Final[tuple[StructuralControlName, ...]] = (
    StructuralControlName.SHORTEST_NOTE_DURATION,
    StructuralControlName.HAS_DOTTED_NOTES,
    StructuralControlName.MAX_NOTES_PER_ONSET,
    StructuralControlName.MAX_NOTES_PER_HAND,
    StructuralControlName.MAX_ONSET_SPAN_SEMITONES,
    StructuralControlName.MAX_MELODIC_GAP_SEMITONES,
    StructuralControlName.STATIC_HAND_SPAN_DEGREES,
    StructuralControlName.BAR_COUNT,
)
