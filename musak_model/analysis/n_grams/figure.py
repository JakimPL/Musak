from fractions import Fraction
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

type FigureDegree = tuple[int, int]
type FigureOnset = tuple[tuple[FigureDegree, ...], Fraction]

_ACCIDENTAL_TEXT: Final[dict[int, str]] = {-1: "b", 0: "", 1: "#"}


class FigureNGram(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    onsets: tuple[FigureOnset, ...] = Field(min_length=1)

    @property
    def n(self) -> int:
        return len(self.onsets)

    def __str__(self) -> str:
        return " ".join(_format_onset(onset) for onset in self.onsets)

    def __repr__(self) -> str:
        return f"FigureNGram({str(self)!r})"


def _format_onset(onset: FigureOnset) -> str:
    degrees, duration = onset
    degree_text = " ".join(_format_degree(degree) for degree in degrees)
    onset_text = degree_text if len(degrees) == 1 else f"[{degree_text}]"
    return f"{onset_text}({duration})"


def _format_degree(degree: FigureDegree) -> str:
    relative_position, accidental = degree
    position_text = f"+{relative_position}" if relative_position > 0 else str(relative_position)
    return f"{position_text}{_ACCIDENTAL_TEXT[accidental]}"
