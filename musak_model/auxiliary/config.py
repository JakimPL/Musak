from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

BOOLEAN_TARGET_CLASS_COUNT: Final[int] = 2


class MusicalAuxiliaryTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    note_density_bucket_boundaries: tuple[float, ...] = Field(min_length=1)
    rhythmic_diversity_bucket_boundaries: tuple[float, ...] = Field(min_length=1)
    voice_independence_bucket_boundaries: tuple[float, ...] = Field(min_length=1)
    hand_span_bucket_boundaries: tuple[int, ...] = Field(min_length=1)

    @field_validator(
        "note_density_bucket_boundaries",
        "rhythmic_diversity_bucket_boundaries",
        "voice_independence_bucket_boundaries",
    )
    @classmethod
    def check_positive_float_boundaries(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(boundary <= 0.0 for boundary in value):
            raise ValueError("bucket boundaries must be positive")

        return _validate_strictly_increasing_float_boundaries(value)

    @field_validator("hand_span_bucket_boundaries")
    @classmethod
    def check_non_negative_integer_boundaries(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(boundary < 0 for boundary in value):
            raise ValueError("bucket boundaries must be non-negative")

        return _validate_strictly_increasing_integer_boundaries(value)

    @property
    def note_density_class_count(self) -> int:
        return len(self.note_density_bucket_boundaries) + 1

    @property
    def rhythmic_diversity_class_count(self) -> int:
        return len(self.rhythmic_diversity_bucket_boundaries) + 1

    @property
    def voice_independence_class_count(self) -> int:
        return len(self.voice_independence_bucket_boundaries) + 1

    @property
    def hand_span_class_count(self) -> int:
        return len(self.hand_span_bucket_boundaries) + 1

    @property
    def uses_accidentals_class_count(self) -> int:
        return BOOLEAN_TARGET_CLASS_COUNT

    @property
    def dotted_duration_class_count(self) -> int:
        return BOOLEAN_TARGET_CLASS_COUNT


def _validate_strictly_increasing_float_boundaries(boundaries: tuple[float, ...]) -> tuple[float, ...]:
    if any(lower >= upper for lower, upper in zip(boundaries, boundaries[1:])):
        raise ValueError("bucket boundaries must be strictly increasing")

    return boundaries


def _validate_strictly_increasing_integer_boundaries(boundaries: tuple[int, ...]) -> tuple[int, ...]:
    if any(lower >= upper for lower, upper in zip(boundaries, boundaries[1:])):
        raise ValueError("bucket boundaries must be strictly increasing")

    return boundaries
