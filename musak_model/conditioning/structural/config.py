from fractions import Fraction

from pydantic import BaseModel, ConfigDict, Field, field_validator

from musak_shared.misc import is_power_of_two


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


class DurationDenominatorBucketConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    thresholds: tuple[int, ...] = Field(min_length=1)

    @field_validator("thresholds", mode="before")
    @classmethod
    def _validate_threshold_items_are_integers(cls, value: object) -> object:
        if isinstance(value, (list, tuple)) and any(
            not isinstance(threshold, int) or isinstance(threshold, bool) for threshold in value
        ):
            raise ValueError("thresholds must be integer denominators")

        return value

    @field_validator("thresholds")
    @classmethod
    def _validate_thresholds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(threshold <= 0 for threshold in value):
            raise ValueError("thresholds must be positive")

        if any(not is_power_of_two(threshold) for threshold in value):
            raise ValueError("thresholds must be powers of two")

        if tuple(sorted(set(value), reverse=True)) != value:
            raise ValueError("thresholds must be unique and sorted from shortest to longest duration")

        return value

    @property
    def duration_thresholds(self) -> tuple[Fraction, ...]:
        return tuple(Fraction(1, threshold) for threshold in self.thresholds)


class StructuralConditioningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shortest_note_duration: DurationDenominatorBucketConfig = DurationDenominatorBucketConfig(
        thresholds=(16, 8, 4, 2, 1)
    )
    max_notes_per_onset: IntegerBucketConfig = IntegerBucketConfig(thresholds=(1, 2, 3, 4))
    max_notes_per_hand: IntegerBucketConfig = IntegerBucketConfig(thresholds=(1, 2, 3, 4, 5))
    max_onset_span_semitones: IntegerBucketConfig = IntegerBucketConfig(thresholds=(3, 7, 12))
    max_melodic_gap_semitones: IntegerBucketConfig = IntegerBucketConfig(thresholds=(2, 4, 7, 12))
    static_hand_span_degrees: IntegerBucketConfig = IntegerBucketConfig(thresholds=(1, 3, 5, 7, 14))
    bar_count: IntegerBucketConfig = IntegerBucketConfig(thresholds=(1, 2, 4, 8, 16, 32))
