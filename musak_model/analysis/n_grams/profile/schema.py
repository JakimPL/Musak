from __future__ import annotations

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


class FigureProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: FigureProfileMetadata
    groups: tuple[FigureProfileGroup, ...]


class FigureSampleCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_index: int = Field(ge=0)
    scale_type: ScaleType
    groups: tuple[FigureProfileGroup, ...]
