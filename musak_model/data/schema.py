from enum import StrEnum
from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from musak_model.common.elements import MIDI_MAX_PITCH, PITCHES_PER_OCTAVE
from musak_model.common.validators import is_power_of_two
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


class ParsedNote(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    midi_pitch: int = Field(ge=0, le=MIDI_MAX_PITCH)
    duration: Fraction = Field(gt=0)
    beat_offset: Fraction = Field(ge=0)


class ParsedRest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    duration: Fraction = Field(gt=0)
    beat_offset: Fraction = Field(ge=0)


class ParsedChord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    midi_pitches: list[int]
    duration: Fraction = Field(gt=0)
    beat_offset: Fraction = Field(ge=0)

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
    key_fifths: int = Field(ge=-7, le=7)
    events: list[ParsedEvent]

    @field_validator("time_denominator")
    @classmethod
    def check_time_denominator(cls, value: int) -> int:
        if not is_power_of_two(value):
            raise ValueError("time_denominator must be a power of two")

        return value


class ParsedScore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    key_root: int = Field(ge=0, lt=PITCHES_PER_OCTAVE)
    key_fifths: int = Field(ge=-7, le=7)
    scale_type: ScaleType
    time_numerator: int = Field(gt=0)
    time_denominator: int
    right_hand_bars: list[ParsedBar]
    left_hand_bars: list[ParsedBar]

    @field_validator("time_denominator")
    @classmethod
    def check_time_denominator(cls, value: int) -> int:
        if not is_power_of_two(value):
            raise ValueError("time_denominator must be a power of two")

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
    TOKENIZATION_ERROR = "tokenization_error"
    AMBIGUOUS_SIMULTANEOUS_DURATION = "ambiguous_simultaneous_duration"
    QUANTIZATION_ERROR = "quantization_error"
    QUANTIZATION_COLLISION = "quantization_collision"


class SegmentMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    key_root: int = Field(ge=0, lt=PITCHES_PER_OCTAVE)
    scale_type: ScaleType
    time_numerator: int = Field(gt=0)
    time_denominator: int
    bar_count: int = Field(ge=0)
    window_start_bar: int = Field(ge=0)
    source_file: Path
    difficulty_level: int | None = Field(None, ge=MIN_DIFFICULTY_LEVEL)
    difficulty_features: DifficultyFeatures | None = None
    eligible_for_training: bool = True
    ineligibility_reasons: frozenset[SegmentIneligibilityReason] = Field(default_factory=frozenset)

    @field_validator("time_denominator")
    @classmethod
    def check_time_denominator(cls, value: int) -> int:
        if not is_power_of_two(value):
            raise ValueError("time_denominator must be a power of two")

        return value


class Segment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    tokens: list[Token] = Field(default_factory=list)
    right_hand_tokens: list[Token] = Field(default_factory=list)
    left_hand_tokens: list[Token] = Field(default_factory=list)
    metadata: SegmentMetadata

    @property
    def key_root(self) -> int:
        return self.metadata.key_root

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
