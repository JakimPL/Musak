from enum import StrEnum
from fractions import Fraction
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from musak_model.tokens.schema import (
    MAX_ACCIDENTAL,
    MAX_DEGREE,
    MAX_OCTAVE_OFFSET,
    MIN_ACCIDENTAL,
    MIN_DEGREE,
    MIN_DIFFICULTY_LEVEL,
    MIN_OCTAVE_OFFSET,
    ScaleType,
    Token,
)
from musak_shared.elements import KEY_FIFTHS_MAX, KEY_FIFTHS_MIN, MIDI_MAX_PITCH, PITCHES_PER_OCTAVE
from musak_shared.time_signature import validate_time_denominator


class TieType(StrEnum):
    START = "start"
    CONTINUE = "continue"
    STOP = "stop"
    PARTIAL = "partial"


class ParsedNote(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    midi_pitch: int = Field(ge=0, le=MIDI_MAX_PITCH)
    duration: Fraction = Field(gt=0)
    beat_offset: Fraction = Field(ge=0)
    tie_type: TieType | None = None


class ParsedRest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    duration: Fraction = Field(gt=0)
    beat_offset: Fraction = Field(ge=0)


class ParsedChord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    midi_pitches: list[int]
    duration: Fraction = Field(gt=0)
    beat_offset: Fraction = Field(ge=0)
    tie_type: TieType | None = None

    @field_validator("midi_pitches")
    @classmethod
    def check_midi_pitches(cls, value: list[int]) -> list[int]:
        if not all(0 <= pitch <= MIDI_MAX_PITCH for pitch in value):
            raise ValueError(f"all midi_pitches must be in [0, {MIDI_MAX_PITCH}]")

        return value


ParsedEvent = ParsedNote | ParsedRest | ParsedChord


class ParsedBar(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    time_numerator: int = Field(gt=0)
    time_denominator: int
    measure_duration: Fraction | None = Field(default=None, ge=0)
    declared_key_fifths: int | None = Field(
        default=None,
        ge=KEY_FIFTHS_MIN,
        le=KEY_FIFTHS_MAX,
        validation_alias=AliasChoices("declared_key_fifths", "key_fifths"),
    )
    events: list[ParsedEvent]

    @field_validator("time_denominator")
    @classmethod
    def check_time_denominator(cls, value: int) -> int:
        validate_time_denominator(value)
        return value


class ParsedScore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    declared_key_fifths: int | None = Field(default=None, ge=KEY_FIFTHS_MIN, le=KEY_FIFTHS_MAX)
    scale_root: int = Field(ge=0, lt=PITCHES_PER_OCTAVE)
    key_fifths: int = Field(ge=KEY_FIFTHS_MIN, le=KEY_FIFTHS_MAX)
    scale_type: ScaleType
    time_numerator: int = Field(gt=0)
    time_denominator: int
    right_hand_bars: list[ParsedBar]
    left_hand_bars: list[ParsedBar]

    @field_validator("time_denominator")
    @classmethod
    def check_time_denominator(cls, value: int) -> int:
        validate_time_denominator(value)

        return value


class PitchDegree(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    degree: int = Field(ge=MIN_DEGREE, le=MAX_DEGREE)
    accidental: int = Field(ge=MIN_ACCIDENTAL, le=MAX_ACCIDENTAL)
    octave_offset: int = Field(ge=MIN_OCTAVE_OFFSET, le=MAX_OCTAVE_OFFSET)


class DifficultyFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_right_hand_span_semitones: int
    max_left_hand_span_semitones: int
    notes_per_beat: float
    rhythmic_diversity: float
    voice_independence: float
    has_accidentals: bool
    has_dotted_notes: bool


class SegmentIneligibilityReason(StrEnum):
    MIXED_TIME_SIGNATURE = "mixed_time_signature"
    KEY_SIGNATURE_CHANGE = "key_signature_change"
    REGISTER_OUT_OF_RANGE = "register_out_of_range"
    OVERLAPPING_EVENTS = "overlapping_events"
    BAR_DURATION_OVERFLOW = "bar_duration_overflow"
    AMBIGUOUS_SIMULTANEOUS_DURATION = "ambiguous_simultaneous_duration"
    QUANTIZATION_ERROR = "quantization_error"
    QUANTIZATION_COLLISION = "quantization_collision"
    PARTIAL_CHORD_TIE = "partial_chord_tie"
    TIE_MISMATCH = "tie_mismatch"
    TIE_CONTINUATION_AT_WINDOW_START = "tie_continuation_at_window_start"
    SILENT_BAR = "silent_bar"
    SILENT_EDGE_BAR = "silent_edge_bar"
    SCALE_MATCH_LOW_CONFIDENCE = "scale_match_low_confidence"
    SCALE_MATCH_NO_PITCHES = "scale_match_no_pitches"


class ScaleMatchDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    declared_key_fifths: int | None = Field(default=None, ge=KEY_FIFTHS_MIN, le=KEY_FIFTHS_MAX)
    in_scale_weight_fraction: float = Field(ge=0, le=1)
    out_of_scale_weight_fraction: float = Field(ge=0, le=1)
    explained_out_of_scale_weight_fraction: float = Field(ge=0, le=1)
    unexplained_out_of_scale_weight_fraction: float = Field(ge=0, le=1)
    best_margin: float = Field(ge=0, le=1)
    observed_pitch_class_count: int = Field(ge=0, le=PITCHES_PER_OCTAVE)
    explanation_pitch_class_count: int = Field(ge=0, le=PITCHES_PER_OCTAVE)
    support_candidate_count: int = Field(ge=0)
    tied_best_candidate_count: int = Field(ge=1)
    declared_match_used: bool
    low_confidence: bool
    ambiguous: bool
    no_pitches: bool


class SpellingContextSource(StrEnum):
    DECLARED_KEY_SIGNATURE = "declared_key_signature"
    SCORE_KEY_FIFTHS = "score_key_fifths"
    PITCH_SET_BASIS = "pitch_set_basis"
    DEFAULT_C_MAJOR = "default_c_major"


class TokenizationContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pitch_set_scale_root: int = Field(ge=0, lt=PITCHES_PER_OCTAVE)
    pitch_set_scale_type: ScaleType
    declared_key_fifths: int | None = Field(default=None, ge=KEY_FIFTHS_MIN, le=KEY_FIFTHS_MAX)
    spelling_key_fifths: int = Field(ge=KEY_FIFTHS_MIN, le=KEY_FIFTHS_MAX)
    spelling_context_source: SpellingContextSource


class SegmentMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    scale_root: int = Field(ge=0, lt=PITCHES_PER_OCTAVE)
    scale_type: ScaleType
    tokenization_context: TokenizationContext
    time_numerator: int = Field(gt=0)
    time_denominator: int
    bar_count: int = Field(ge=0)
    bar_durations: tuple[Fraction, ...] | None = None
    window_start_bar: int = Field(ge=0)
    source_file: Path
    difficulty_level: int | None = Field(None, ge=MIN_DIFFICULTY_LEVEL)
    difficulty_features: DifficultyFeatures | None = None
    scale_match: ScaleMatchDiagnostics | None = None
    eligible_for_training: bool = True
    ineligibility_reasons: frozenset[SegmentIneligibilityReason] = Field(default_factory=frozenset)

    @field_validator("time_denominator")
    @classmethod
    def check_time_denominator(cls, value: int) -> int:
        validate_time_denominator(value)
        return value


class Segment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    tokens: list[Token]
    metadata: SegmentMetadata

    @property
    def scale_root(self) -> int:
        return self.metadata.scale_root

    @property
    def scale_type(self) -> ScaleType:
        return self.metadata.scale_type

    @property
    def time_numerator(self) -> int:
        return self.metadata.time_numerator

    @property
    def time_denominator(self) -> int:
        return self.metadata.time_denominator

    @property
    def bar_count(self) -> int:
        return self.metadata.bar_count

    @property
    def source_file(self) -> Path:
        return self.metadata.source_file

    @property
    def difficulty_level(self) -> int | None:
        return self.metadata.difficulty_level

    @property
    def difficulty_features(self) -> DifficultyFeatures | None:
        return self.metadata.difficulty_features
