import math
from fractions import Fraction
from functools import cached_property
from typing import Any, Iterator, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, computed_field, field_validator

from modules.rhythm.exceptions import EmptyScoreException
from modules.rhythm.note import Note, NoteType
from modules.rhythm.time_signature import TimeSignatureType


class Phrase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    notes: list[Note] = []

    def __init__(self, notes: Optional[Sequence[NoteType]] = None, **data: Any) -> None:
        if notes is not None:
            super().__init__(notes=list(notes), **data)
        else:
            super().__init__(**data)

    @field_validator("notes", mode="before")
    @classmethod
    def _coerce_notes(cls, v: object) -> list[Note]:
        if isinstance(v, (list, tuple)):
            return [n if isinstance(n, Note) else Note(n) for n in v]  # type: ignore[arg-type]
        raise TypeError(f"expected a list or tuple, got {type(v)}")

    def __add__(self, other: "Phrase") -> "Phrase":
        if not isinstance(other, Phrase):
            raise TypeError(f"cannot add phrase to {type(other)}")
        return Phrase(self.notes + other.notes)

    def __bool__(self) -> bool:
        return bool(self.notes)

    def __iter__(self) -> Iterator[Note]:
        return iter(self.notes)

    def __len__(self) -> int:
        return len(self.notes)

    def validate(self, time_signature: TimeSignatureType = (4, 4)) -> int:
        if not self.notes:
            raise EmptyScoreException("an empty phrase")

        total_length: Fraction = Fraction(0)
        checkpoints: list[Fraction] = [Fraction(0)]

        for note in self.notes:
            total_length += note.duration
            checkpoints.append(total_length)

        time_signature_fraction = Fraction(*time_signature)
        validation_set = {
            index * time_signature_fraction
            for index in range(math.ceil(total_length / time_signature_fraction) + 1)
        }
        difference = validation_set.difference(set(checkpoints))

        return int(min(difference) / time_signature_fraction) if difference else 0

    @computed_field  # type: ignore[misc]
    @cached_property
    def length(self) -> Fraction:
        return sum((note.duration for note in self.notes), Fraction(0))

    @computed_field  # type: ignore[misc]
    @cached_property
    def lcm(self) -> int:
        return math.lcm(*(note.duration.denominator for note in self.notes))


PhraseType = Union[Phrase, Sequence[NoteType]]
