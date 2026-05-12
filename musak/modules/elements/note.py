from __future__ import annotations

from fractions import Fraction
from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field, model_validator

from musak.modules.elements.exceptions import NoteNotSupportedError
from musak.modules.elements.misc import is_power_of_two


class Note(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    duration: Fraction = Field(gt=0)
    pause: bool = Field(default=False)

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(
        cls,
        data: int | tuple[int, int] | Fraction | Note | dict[str, Fraction | bool],
    ) -> dict[str, Fraction | bool]:
        if isinstance(data, Note):
            return {"duration": data.duration, "pause": data.pause}

        if isinstance(data, int):
            if data == 0:
                raise ValueError("duration value cannot be zero")

            return {"duration": Fraction(1, abs(data)), "pause": data < 0}

        if isinstance(data, Fraction):
            return {"duration": abs(data), "pause": data < 0}

        if isinstance(data, tuple):
            numerator, denominator = data
            return {
                "duration": abs(Fraction(numerator, denominator)),
                "pause": numerator * denominator < 0,
            }

        raw_duration = data.get("duration")
        if isinstance(raw_duration, int):
            if raw_duration == 0:
                raise ValueError("duration value cannot be zero")

            return {
                "duration": Fraction(1, abs(raw_duration)),
                "pause": data.get("pause", raw_duration < 0),
            }

        if isinstance(raw_duration, tuple):
            numerator, denominator = raw_duration
            return {
                "duration": abs(Fraction(numerator, denominator)),
                "pause": data.get("pause", numerator * denominator < 0),
            }

        return data

    @model_validator(mode="after")
    def _validate_note(self) -> Note:
        numerator, denominator, _ = self.dots
        if numerator != 1 or denominator == 0 or not is_power_of_two(denominator):
            raise NoteNotSupportedError(f"note {self.duration} is not supported")

        return self

    @cached_property
    def dots(self) -> tuple[int, int, int]:
        numerator = self.duration.numerator
        denominator = self.duration.denominator
        dots = 0
        while numerator % 3 == 0 and denominator % 2 == 0:
            numerator //= 3
            denominator //= 2
            dots += 1

        return numerator, denominator, dots

    def __repr__(self) -> str:
        _, denominator, dots = self.dots
        return f"{'r' if self.pause else 'c'}{denominator}{'.' * dots}"

    def __str__(self) -> str:
        return self.__repr__()


NoteType = int | tuple[int, int] | Note
