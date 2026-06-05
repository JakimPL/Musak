from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from musak_model.tokens.schema import Hand
from musak_shared.time_signature import validate_time_denominator

DEFAULT_RHYTHM_GRID_DENOMINATOR: Final[int] = 16


class RhythmCellState(StrEnum):
    UNKNOWN = "unknown"
    REST = "rest"
    ONSET = "onset"
    SUSTAIN = "sustain"


class CoactivityState(StrEnum):
    SILENT = "silent"
    RIGHT_ONLY = "right_only"
    LEFT_ONLY = "left_only"
    BOTH_SYNCHRONIZED = "both_synchronized"
    RIGHT_ONSET_LEFT_SUSTAIN = "right_onset_left_sustain"
    LEFT_ONSET_RIGHT_SUSTAIN = "left_onset_right_sustain"
    BOTH_SUSTAIN = "both_sustain"
    BOTH_ACTIVE = "both_active"


class RhythmGridConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    grid_denominator: int = Field(default=DEFAULT_RHYTHM_GRID_DENOMINATOR, gt=0)


class RhythmGridCell(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    global_cell_index: int = Field(ge=0)
    bar_index: int = Field(ge=0)
    cell_index: int = Field(ge=0)
    start: Fraction = Field(ge=0)
    end: Fraction = Field(gt=0)
    bar_relative_start: Fraction = Field(ge=0)
    bar_relative_end: Fraction = Field(gt=0)
    metrical_offset: Fraction = Field(ge=0)
    distance_to_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if self.end <= self.start:
            raise ValueError("rhythm grid cell end must be greater than start")
        if self.bar_relative_end <= self.bar_relative_start:
            raise ValueError("rhythm grid cell bar-relative end must be greater than start")
        return self


class RhythmGridFrame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    config: RhythmGridConfig
    time_numerator: int = Field(gt=0)
    time_denominator: int
    bar_durations: tuple[Fraction, ...]
    cells: tuple[RhythmGridCell, ...]
    right_hand_states: tuple[RhythmCellState, ...]
    left_hand_states: tuple[RhythmCellState, ...]
    coactivity_states: tuple[CoactivityState, ...]

    @field_validator("time_denominator")
    @classmethod
    def _validate_time_denominator(cls, value: int) -> int:
        validate_time_denominator(value)
        return value

    @model_validator(mode="after")
    def _validate_lengths(self) -> Self:
        cell_count = len(self.cells)
        if len(self.right_hand_states) != cell_count:
            raise ValueError("right-hand state count must match rhythm grid cell count")
        if len(self.left_hand_states) != cell_count:
            raise ValueError("left-hand state count must match rhythm grid cell count")
        if len(self.coactivity_states) != cell_count:
            raise ValueError("coactivity state count must match rhythm grid cell count")
        return self

    def states_for_hand(self, hand: Hand) -> tuple[RhythmCellState, ...]:
        match hand:
            case Hand.RIGHT:
                return self.right_hand_states
            case Hand.LEFT:
                return self.left_hand_states
        raise ValueError(f"unsupported hand: {hand}")
