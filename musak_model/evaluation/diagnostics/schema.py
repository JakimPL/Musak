from pydantic import BaseModel, ConfigDict, Field


class SegmentDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    right_silence_fraction: float = Field(ge=0)
    left_silence_fraction: float = Field(ge=0)
    both_hands_silence_fraction: float = Field(ge=0)
    both_hands_active_fraction: float = Field(ge=0)
    right_only_active_fraction: float = Field(ge=0)
    left_only_active_fraction: float = Field(ge=0)
    longest_right_silence_beats: float = Field(ge=0)
    longest_left_silence_beats: float = Field(ge=0)
    longest_both_hands_silence_beats: float = Field(ge=0)
    right_note_onsets_per_bar: float = Field(ge=0)
    left_note_onsets_per_bar: float = Field(ge=0)
    silent_bar_count: int = Field(ge=0)
    silent_bar_fraction: float = Field(ge=0, le=1)
    silent_edge_bar_count: int = Field(ge=0)
    hand_activity_balance: float = Field(ge=0, le=1)
    empty_score: bool
    one_hand_only: bool
    note_token_fraction: float = Field(ge=0, le=1)
    rest_token_fraction: float = Field(ge=0, le=1)
    hold_token_fraction: float = Field(ge=0, le=1)
    accidental_note_fraction: float = Field(ge=0, le=1)
    in_scale_note_fraction: float = Field(ge=0, le=1)
    note_density_per_beat: float = Field(ge=0)
    onset_density_per_beat: float = Field(ge=0)
    right_onset_density_per_beat: float = Field(ge=0)
    left_onset_density_per_beat: float = Field(ge=0)
    shortest_note_duration_beats: float = Field(ge=0)
    has_dotted_notes: bool
    max_notes_per_onset: int = Field(ge=0)
    max_notes_per_hand: int = Field(ge=0)
    max_onset_span_semitones: int = Field(ge=0)
    max_melodic_gap_semitones: int = Field(ge=0)
    static_hand_span_degrees: int = Field(ge=0)
    synchronized_onset_fraction: float = Field(ge=0, le=1)
    independent_onset_fraction: float = Field(ge=0, le=1)

    def to_manifest_values(self) -> dict[str, float | bool]:
        return self.model_dump()
