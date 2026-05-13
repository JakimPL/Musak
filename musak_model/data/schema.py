from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from musak_model.common.elements import PITCHES_PER_OCTAVE, VALID_MODES
from musak_model.common.validators import is_power_of_two
from musak_model.tokens.schema import (
    MAX_DEGREE,
    MAX_DIFFICULTY_LEVEL,
    MIN_DEGREE,
    MIN_DIFFICULTY_LEVEL,
    ScaleType,
    Token,
)


class ParsedNote(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    midi_pitch: int
    duration: Fraction
    beat_offset: Fraction

    @field_validator("midi_pitch")
    @classmethod
    def check_midi_pitch(cls, value: int) -> int:
        if not 0 <= value <= 127:
            raise ValueError("midi_pitch must be in [0, 127]")

        return value

    @field_validator("duration", "beat_offset")
    @classmethod
    def check_positive_fraction(cls, value: Fraction) -> Fraction:
        if value <= 0:
            raise ValueError("duration and beat_offset must be positive")

        return value


class ParsedRest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    duration: Fraction
    beat_offset: Fraction

    @field_validator("duration", "beat_offset")
    @classmethod
    def check_positive_fraction(cls, value: Fraction) -> Fraction:
        if value <= 0:
            raise ValueError("duration and beat_offset must be positive")

        return value


class ParsedChord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    midi_pitches: list[int]
    duration: Fraction
    beat_offset: Fraction

    @field_validator("midi_pitches")
    @classmethod
    def check_midi_pitches(cls, value: list[int]) -> list[int]:
        if not all(0 <= pitch <= 127 for pitch in value):
            raise ValueError("all midi_pitches must be in [0, 127]")

        return value

    @field_validator("duration", "beat_offset")
    @classmethod
    def check_positive_fraction(cls, value: Fraction) -> Fraction:
        if value <= 0:
            raise ValueError("duration and beat_offset must be positive")

        return value


ParsedEvent = ParsedNote | ParsedRest | ParsedChord


class ParsedBar(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    events: list[ParsedEvent]


class ParsedScore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    key_root: int
    key_fifths: int
    mode: str
    time_numerator: int
    time_denominator: int
    right_hand_bars: list[ParsedBar]
    left_hand_bars: list[ParsedBar]

    @field_validator("mode")
    @classmethod
    def check_mode(cls, value: str) -> str:
        if value not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")

        return value

    @field_validator("key_fifths")
    @classmethod
    def check_key_fifths(cls, value: int) -> int:
        # MusicXML and music21 allow -7 (Cb) to +7 (C#)
        if not -7 <= value <= 7:
            raise ValueError("key_fifths must be in [-7, 7]")

        return value

    @field_validator("key_root")
    @classmethod
    def check_key_root(cls, value: int) -> int:
        if not 0 <= value < PITCHES_PER_OCTAVE:
            raise ValueError(f"key_root must be in [0, {PITCHES_PER_OCTAVE - 1}]")

        return value

    @field_validator("time_numerator")
    @classmethod
    def check_time_numerator(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("time_numerator must be positive")

        return value

    @field_validator("time_denominator")
    @classmethod
    def check_time_denominator(cls, value: int) -> int:
        if not is_power_of_two(value):
            raise ValueError("time_denominator must be a power of two")

        return value


class PitchDegree(BaseModel):
    model_config = ConfigDict(frozen=True)

    degree: int
    accidental: int
    octave_offset: int

    @field_validator("degree")
    @classmethod
    def check_degree(cls, value: int) -> int:
        if not MIN_DEGREE <= value <= MAX_DEGREE:
            raise ValueError(f"degree must be in [{MIN_DEGREE}, {MAX_DEGREE}]")

        return value


class DifficultyFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_right_hand_span_semitones: int
    max_left_hand_span_semitones: int
    notes_per_beat: float
    rhythmic_diversity: float
    voice_independence: float
    has_accidentals: bool
    has_dotted_notes: bool


class SegmentMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    key_root: int
    scale_type: ScaleType
    time_numerator: int
    time_denominator: int
    bar_count: int
    source_file: Path
    difficulty_level: int | None = None
    difficulty_features: DifficultyFeatures | None = None

    @field_validator("key_root")
    @classmethod
    def check_key_root(cls, value: int) -> int:
        if not 0 <= value < PITCHES_PER_OCTAVE:
            raise ValueError(f"key_root must be in [0, {PITCHES_PER_OCTAVE - 1}]")

        return value

    @field_validator("time_numerator")
    @classmethod
    def check_time_numerator(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("time_numerator must be positive")

        return value

    @field_validator("time_denominator")
    @classmethod
    def check_time_denominator(cls, value: int) -> int:
        if not is_power_of_two(value):
            raise ValueError("time_denominator must be a power of two")

        return value

    @field_validator("bar_count")
    @classmethod
    def check_bar_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("bar_count must be non-negative")

        return value

    @field_validator("difficulty_level")
    @classmethod
    def check_difficulty_level(cls, value: int | None) -> int | None:
        if value is not None and not MIN_DIFFICULTY_LEVEL <= value <= MAX_DIFFICULTY_LEVEL:
            raise ValueError(f"difficulty_level must be in [{MIN_DIFFICULTY_LEVEL}, {MAX_DIFFICULTY_LEVEL}]")

        return value


class Segment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    right_hand_tokens: list[Token]
    left_hand_tokens: list[Token]
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
