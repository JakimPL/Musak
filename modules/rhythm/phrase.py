from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from functools import cached_property

from pydantic import BaseModel, ConfigDict, field_validator

from modules.rhythm.exceptions import EmptyScoreException
from modules.rhythm.note import Note, NoteType
from modules.rhythm.time_signature import DEFAULT_TIME_SIGNATURE, TimeSignatureType


class Phrase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    notes: list[Note] = []

    @field_validator("notes", mode="before")
    @classmethod
    def _coerce_notes(cls, v: Sequence[NoteType] | None) -> list[Note]:
        if v is None:
            return []
        result: list[Note] = []
        for note in v:
            result.append(Note.model_validate(note))

        return result

    def __add__(self, other: Phrase) -> Phrase:
        if not isinstance(other, Phrase):
            raise TypeError(f"cannot add phrase to {type(other)}")
        return Phrase(notes=self.notes + other.notes)

    def __bool__(self) -> bool:
        return bool(self.notes)

    def __len__(self) -> int:
        return len(self.notes)

    def find_invalid_beat(
        self, time_signature: TimeSignatureType = DEFAULT_TIME_SIGNATURE
    ) -> int:
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

    @cached_property
    def length(self) -> Fraction:
        return sum((note.duration for note in self.notes), Fraction(0))

    @cached_property
    def lcm(self) -> int:
        return math.lcm(*(note.duration.denominator for note in self.notes))


PhraseType = Phrase | Sequence[NoteType]
