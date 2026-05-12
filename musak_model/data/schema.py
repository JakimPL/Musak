from fractions import Fraction

from pydantic import BaseModel, ConfigDict

from musak_model.tokens.schema import ScaleType, Token


class ParsedNote(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    midi_pitch: int
    duration: Fraction
    beat_offset: Fraction


class ParsedRest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    duration: Fraction
    beat_offset: Fraction


class ParsedChord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    midi_pitches: list[int]
    duration: Fraction
    beat_offset: Fraction


ParsedEvent = ParsedNote | ParsedRest | ParsedChord


class ParsedBar(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    events: list[ParsedEvent]


class ParsedScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    key_root: int
    mode: str
    time_numerator: int
    time_denominator: int
    right_hand_bars: list[ParsedBar]
    left_hand_bars: list[ParsedBar]


class PitchDegree(BaseModel):
    model_config = ConfigDict(frozen=True)

    degree: int
    accidental: int
    octave_offset: int


class DifficultyFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_right_hand_span_semitones: int
    max_left_hand_span_semitones: int
    notes_per_beat: float
    rhythmic_diversity: float
    voice_independence: float
    has_accidentals: bool
    has_dotted_notes: bool


class Segment(BaseModel):
    model_config = ConfigDict(frozen=True)

    right_hand_tokens: list[Token]
    left_hand_tokens: list[Token]
    key_root: int
    scale_type: ScaleType
    time_numerator: int
    time_denominator: int
    bar_count: int
    source_file: str
    difficulty_level: int | None = None
    difficulty_features: DifficultyFeatures | None = None
