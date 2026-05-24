from fractions import Fraction

from pydantic import BaseModel, ConfigDict, Field

type FigureDegree = tuple[int, int]
type FigureOnset = tuple[tuple[FigureDegree, ...], Fraction]


class FigureNGram(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    onsets: tuple[FigureOnset, ...] = Field(min_length=1)

    @property
    def n(self) -> int:
        return len(self.onsets)
