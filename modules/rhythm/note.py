from fractions import Fraction
from functools import cached_property
from typing import Any, Tuple, Union

from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from modules.rhythm.exceptions import NoteNotSupportedError
from modules.rhythm.misc import is_power_of_two


class Note(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    duration: Fraction
    pause: bool = False

    def __init__(
        self,
        argument: Union[int, Tuple[int, int], Fraction, "Note", None] = None,
        **data: Any,
    ) -> None:
        if argument is not None:
            if isinstance(argument, Note):
                super().__init__(duration=argument.duration, pause=argument.pause)
            elif isinstance(argument, int):
                if argument == 0:
                    raise ValueError("a value has to be non-zero")
                super().__init__(
                    duration=Fraction(1, abs(argument)), pause=argument < 0
                )
            elif isinstance(argument, Fraction):
                super().__init__(duration=abs(argument), pause=argument < 0)
            elif isinstance(argument, tuple):
                if len(argument) == 2 and all(isinstance(e, int) for e in argument):
                    super().__init__(
                        duration=abs(Fraction(*argument)),
                        pause=(argument[0] * argument[1] < 0),
                    )
                else:
                    raise ValueError(
                        f"invalid tuple: {argument}, expected tuple[int, int]"
                    )
            else:
                raise TypeError(
                    f"expected int, tuple[int, int] or Fraction, got {type(argument)}"
                )
        else:
            super().__init__(**data)

    @model_validator(mode="after")
    def _validate_note(self) -> "Note":
        numerator, denominator, _ = self.dots
        if numerator != 1 or denominator == 0 or not is_power_of_two(denominator):
            raise NoteNotSupportedError(f"note {self.duration} is not supported")
        return self

    @computed_field  # type: ignore[misc]
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
        numerator, denominator, dots = self.dots
        return "{type}{length}{dotted}".format(
            type="r" if self.pause else "c",
            length=denominator,
            dotted="." * dots,
        )


NoteType = Union[int, Tuple[int, int], Note]
