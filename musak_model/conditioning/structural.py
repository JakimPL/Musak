from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from musak_model.data.schema import Segment
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import note_token_to_midi_pitch, note_token_to_static_hand_position
from musak_model.tokens.schema import Hand, HandToken, JoinWithPreviousToken, NoteToken
from musak_shared.elements import is_dotted_duration
from musak_shared.ratios import parse_ratio

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


class IntegerBucketConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    thresholds: tuple[int, ...] = Field(min_length=1)

    @field_validator("thresholds")
    @classmethod
    def _validate_thresholds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(threshold < 0 for threshold in value):
            raise ValueError("thresholds must be non-negative")

        if tuple(sorted(set(value))) != value:
            raise ValueError("thresholds must be unique and sorted")

        return value


class FractionBucketConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    thresholds: tuple[str, ...] = Field(min_length=1)

    @field_validator("thresholds")
    @classmethod
    def _validate_thresholds(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        parsed = tuple(parse_ratio(threshold) for threshold in value)
        if any(threshold <= 0 for threshold in parsed):
            raise ValueError("thresholds must be positive")

        if tuple(sorted(set(parsed))) != parsed:
            raise ValueError("thresholds must be unique and sorted")

        return value

    @property
    def parsed_thresholds(self) -> tuple[Fraction, ...]:
        return tuple(parse_ratio(threshold) for threshold in self.thresholds)


class StructuralConditioningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shortest_note_duration: FractionBucketConfig = FractionBucketConfig(thresholds=("1/16", "1/8", "1/4", "1/2", "1/1"))
    max_notes_per_onset: IntegerBucketConfig = IntegerBucketConfig(thresholds=(1, 2, 3, 4))
    max_notes_per_hand: IntegerBucketConfig = IntegerBucketConfig(thresholds=(1, 2, 3, 4, 5))
    max_onset_span_semitones: IntegerBucketConfig = IntegerBucketConfig(thresholds=(3, 7, 12))
    max_melodic_gap_semitones: IntegerBucketConfig = IntegerBucketConfig(thresholds=(2, 4, 7, 12))
    static_hand_span_degrees: IntegerBucketConfig = IntegerBucketConfig(thresholds=(1, 3, 5, 7, 14))
    bar_count: IntegerBucketConfig = IntegerBucketConfig(thresholds=(1, 2, 4, 8, 16, 32))


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


class StructuralControlVocabulary:
    def __init__(self, config: StructuralConditioningConfig) -> None:
        self._config = config

    @property
    def vocabulary_sizes(self) -> tuple[int, ...]:
        return tuple(self.vocabulary_size(control_name) for control_name in STRUCTURAL_CONTROL_ORDER)

    def control_index(self, control_name: StructuralControlName) -> int:
        return STRUCTURAL_CONTROL_ORDER.index(control_name)

    def vocabulary_size(self, control_name: StructuralControlName) -> int:
        match control_name:
            case StructuralControlName.SHORTEST_NOTE_DURATION:
                return _bucket_size(self._config.shortest_note_duration.parsed_thresholds)
            case StructuralControlName.HAS_DOTTED_NOTES:
                return BOOLEAN_CONTROL_VOCABULARY_SIZE
            case StructuralControlName.MAX_NOTES_PER_ONSET:
                return _bucket_size(self._config.max_notes_per_onset.thresholds)
            case StructuralControlName.MAX_NOTES_PER_HAND:
                return _bucket_size(self._config.max_notes_per_hand.thresholds)
            case StructuralControlName.MAX_ONSET_SPAN_SEMITONES:
                return _bucket_size(self._config.max_onset_span_semitones.thresholds)
            case StructuralControlName.MAX_MELODIC_GAP_SEMITONES:
                return _bucket_size(self._config.max_melodic_gap_semitones.thresholds)
            case StructuralControlName.STATIC_HAND_SPAN_DEGREES:
                return _bucket_size(self._config.static_hand_span_degrees.thresholds)
            case StructuralControlName.BAR_COUNT:
                return _bucket_size(self._config.bar_count.thresholds)

    def features_to_ids(self, features: StructuralControlFeatures | None) -> tuple[int, ...]:
        if features is None:
            return tuple(UNKNOWN_CONTROL_ID for _ in STRUCTURAL_CONTROL_ORDER)

        return (
            _fraction_bucket_id(features.shortest_note_duration, self._config.shortest_note_duration.parsed_thresholds),
            _boolean_control_id(features.has_dotted_notes),
            _integer_bucket_id(features.max_notes_per_onset, self._config.max_notes_per_onset.thresholds),
            _integer_bucket_id(features.max_notes_per_hand, self._config.max_notes_per_hand.thresholds),
            _integer_bucket_id(features.max_onset_span_semitones, self._config.max_onset_span_semitones.thresholds),
            _integer_bucket_id(features.max_melodic_gap_semitones, self._config.max_melodic_gap_semitones.thresholds),
            _integer_bucket_id(features.static_hand_span_degrees, self._config.static_hand_span_degrees.thresholds),
            _integer_bucket_id(features.bar_count, self._config.bar_count.thresholds),
        )


def extract_structural_control_features(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> StructuralControlFeatures:
    state = _FeatureState()
    tokens = segment.tokens
    active_hand = Hand.RIGHT

    for index, token in enumerate(tokens):
        match token:
            case HandToken():
                active_hand = token.hand
            case NoteToken():
                next_token = tokens[index + 1] if index + 1 < len(tokens) else None
                joins_previous_onset = isinstance(next_token, JoinWithPreviousToken)
                state = state.add_note(
                    token,
                    hand=active_hand,
                    joins_previous_onset=joins_previous_onset,
                    segment=segment,
                    duration_vocabulary=duration_vocabulary,
                )
            case JoinWithPreviousToken():
                continue
            case _:
                continue

    return state.to_features()


class _FeatureState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shortest_note_duration: Fraction | None = None
    has_dotted_notes: bool = False
    max_notes_per_onset: int = 0
    max_onset_span_semitones: int = 0
    max_melodic_gap_semitones: int = 0
    right_static_positions: tuple[int, ...] = ()
    left_static_positions: tuple[int, ...] = ()
    right_last_onset_pitches: tuple[int, ...] = ()
    left_last_onset_pitches: tuple[int, ...] = ()
    right_current_onset_count: int = 0
    left_current_onset_count: int = 0

    def add_note(
        self,
        token: NoteToken,
        *,
        hand: Hand,
        joins_previous_onset: bool,
        segment: Segment,
        duration_vocabulary: DurationVocabulary,
    ) -> _FeatureState:
        duration = duration_vocabulary.id_to_fraction(token.duration_id)
        midi_pitch = note_token_to_midi_pitch(
            token,
            key_root=segment.key_root,
            scale_type=segment.scale_type,
            hand=hand,
        )
        state = self.model_copy(
            update={
                "shortest_note_duration": _minimum_optional_fraction(self.shortest_note_duration, duration),
                "has_dotted_notes": self.has_dotted_notes or is_dotted_duration(duration),
            }
        )
        state = state.add_static_position(note_token_to_static_hand_position(token), hand=hand)
        if joins_previous_onset:
            return state.add_chord_note(midi_pitch, hand=hand)

        return state.add_new_onset(midi_pitch, hand=hand)

    def to_features(self) -> StructuralControlFeatures:
        return StructuralControlFeatures(
            shortest_note_duration=self.shortest_note_duration,
            has_dotted_notes=self.has_dotted_notes,
            max_notes_per_onset=self.max_notes_per_onset,
            max_notes_per_hand=self.max_notes_per_onset,
            max_onset_span_semitones=self.max_onset_span_semitones,
            max_melodic_gap_semitones=self.max_melodic_gap_semitones,
            static_hand_span_degrees=max(
                _inclusive_span(self.right_static_positions),
                _inclusive_span(self.left_static_positions),
            ),
            bar_count=None,
        )

    def add_static_position(self, position: int, *, hand: Hand) -> _FeatureState:
        match hand:
            case Hand.RIGHT:
                return self.model_copy(update={"right_static_positions": (*self.right_static_positions, position)})
            case Hand.LEFT:
                return self.model_copy(update={"left_static_positions": (*self.left_static_positions, position)})

    def add_chord_note(self, midi_pitch: int, *, hand: Hand) -> _FeatureState:
        match hand:
            case Hand.RIGHT:
                count = self.right_current_onset_count + 1
                current_pitches = (*self.right_last_onset_pitches, midi_pitch)
                return self.model_copy(
                    update={
                        "right_current_onset_count": count,
                        "right_last_onset_pitches": current_pitches,
                        "max_notes_per_onset": max(self.max_notes_per_onset, count),
                        "max_onset_span_semitones": max(
                            self.max_onset_span_semitones,
                            _distance_span(current_pitches),
                        ),
                    }
                )
            case Hand.LEFT:
                count = self.left_current_onset_count + 1
                current_pitches = (*self.left_last_onset_pitches, midi_pitch)
                return self.model_copy(
                    update={
                        "left_current_onset_count": count,
                        "left_last_onset_pitches": current_pitches,
                        "max_notes_per_onset": max(self.max_notes_per_onset, count),
                        "max_onset_span_semitones": max(
                            self.max_onset_span_semitones,
                            _distance_span(current_pitches),
                        ),
                    }
                )

    def add_new_onset(self, midi_pitch: int, *, hand: Hand) -> _FeatureState:
        match hand:
            case Hand.RIGHT:
                return self.model_copy(
                    update={
                        "right_current_onset_count": 1,
                        "right_last_onset_pitches": (midi_pitch,),
                        "max_notes_per_onset": max(self.max_notes_per_onset, 1),
                        "max_melodic_gap_semitones": max(
                            self.max_melodic_gap_semitones,
                            _melodic_gap(midi_pitch, self.right_last_onset_pitches),
                        ),
                    }
                )
            case Hand.LEFT:
                return self.model_copy(
                    update={
                        "left_current_onset_count": 1,
                        "left_last_onset_pitches": (midi_pitch,),
                        "max_notes_per_onset": max(self.max_notes_per_onset, 1),
                        "max_melodic_gap_semitones": max(
                            self.max_melodic_gap_semitones,
                            _melodic_gap(midi_pitch, self.left_last_onset_pitches),
                        ),
                    }
                )


def _minimum_optional_fraction(current: Fraction | None, candidate: Fraction) -> Fraction:
    if current is None:
        return candidate

    return min(current, candidate)


def _melodic_gap(midi_pitch: int, previous_onset_pitches: tuple[int, ...]) -> int:
    if not previous_onset_pitches:
        return 0

    return min(abs(midi_pitch - previous_pitch) for previous_pitch in previous_onset_pitches)


def _inclusive_span(values: tuple[int, ...]) -> int:
    if len(values) < 2:
        return 0

    return max(values) - min(values) + 1


def _distance_span(values: tuple[int, ...]) -> int:
    if len(values) < 2:
        return 0

    return max(values) - min(values)


def _bucket_size(thresholds: tuple[object, ...]) -> int:
    return len(thresholds) + 2


def _integer_bucket_id(value: int | None, thresholds: tuple[int, ...]) -> int:
    if value is None:
        return UNKNOWN_CONTROL_ID

    for index, threshold in enumerate(thresholds, start=1):
        if value <= threshold:
            return index

    return len(thresholds) + 1


def _boolean_control_id(value: bool | None) -> int:
    if value is None:
        return UNKNOWN_CONTROL_ID

    return TRUE_CONTROL_ID if value else FALSE_CONTROL_ID


def _fraction_bucket_id(value: Fraction | None, thresholds: tuple[Fraction, ...]) -> int:
    if value is None:
        return UNKNOWN_CONTROL_ID

    for index, threshold in enumerate(thresholds, start=1):
        if value <= threshold:
            return index

    return len(thresholds) + 1
