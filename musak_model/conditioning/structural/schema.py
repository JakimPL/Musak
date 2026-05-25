from fractions import Fraction

from pydantic import BaseModel, ConfigDict


class StructuralControlFeatures(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shortest_note_duration: Fraction | None
    has_dotted_notes: bool | None
    max_notes_per_onset: int | None
    max_notes_per_hand: int | None
    max_onset_span_semitones: int | None
    max_melodic_gap_semitones: int | None
    static_hand_span_degrees: int | None
    bar_count: int | None
