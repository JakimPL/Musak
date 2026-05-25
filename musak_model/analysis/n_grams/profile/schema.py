from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from musak_model.tokens.schema import Hand, ScaleType


class FigureProfileMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_n: int = Field(gt=0)
    max_n: int = Field(gt=0)
    sample_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_n_range(self) -> FigureProfileMetadata:
        if self.max_n < self.min_n:
            raise ValueError("max_n must be greater than or equal to min_n")

        return self


class FigureProfileGroup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scale_type: ScaleType
    hand: Hand
    n: int = Field(gt=0)
    total: int = Field(ge=0)
    monophonic: int = Field(ge=0)
    chords_only: int = Field(ge=0)
    in_scale: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_property_totals(self) -> Self:
        if self.monophonic > self.total:
            raise ValueError("monophonic must be less than or equal to total")

        if self.chords_only > self.total:
            raise ValueError("chords_only must be less than or equal to total")

        if self.in_scale > self.total:
            raise ValueError("in_scale must be less than or equal to total")

        return self


class FigureProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: FigureProfileMetadata
    groups: tuple[FigureProfileGroup, ...]

    @model_validator(mode="after")
    def _validate_group_n_range(self) -> Self:
        invalid_n_values = sorted(
            {group.n for group in self.groups if group.n < self.metadata.min_n or group.n > self.metadata.max_n}
        )
        if invalid_n_values:
            raise ValueError("group n values must fall within metadata min_n/max_n range; " f"found {invalid_n_values}")

        return self


class FigureSampleCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_index: int = Field(ge=0)
    scale_type: ScaleType
    groups: tuple[FigureProfileGroup, ...]

    @model_validator(mode="after")
    def _validate_group_scale_types(self) -> Self:
        invalid_scale_types = sorted(
            {group.scale_type.value for group in self.groups if group.scale_type != self.scale_type}
        )
        if invalid_scale_types:
            raise ValueError(
                "sample count group scale_type values must match the sample scale_type; " f"found {invalid_scale_types}"
            )

        return self
