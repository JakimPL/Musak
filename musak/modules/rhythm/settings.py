from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from musak.config.defaults import TIME_SIGNATURE
from musak.modules.elements.note import Note, NoteType
from musak.modules.elements.phrase import Phrase, PhraseType
from musak_shared.time_signature import TimeSignatureType, validate_time_signature


class GroupSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    notes: list[Note] = []
    phrases: list[Phrase] = []

    @field_validator("notes", mode="before")
    @classmethod
    def _coerce_notes(cls, v: Sequence[NoteType] | None) -> list[Note]:
        if v is None:
            return []

        result: list[Note] = []
        for note in v:
            result.append(Note.model_validate(note))

        return result

    @field_validator("phrases", mode="before")
    @classmethod
    def _coerce_phrases(cls, v: Sequence[PhraseType] | None) -> list[Phrase]:
        if v is None:
            return []

        result: list[Phrase] = []
        for phrase in v:
            result.append(Phrase.model_validate(phrase))

        return result

    @model_validator(mode="after")
    def _validate_non_empty(self) -> GroupSettings:
        if not self.get_all_phrases():
            raise ValueError("notes and phrases cannot be both empty")

        return self

    def get_all_phrases(self) -> list[Phrase]:
        return [Phrase(notes=[note]) for note in self.notes] + self.phrases


class Settings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    time_signature: TimeSignatureType = TIME_SIGNATURE
    tempo: int = Field(gt=0)
    groups: int = Field(ge=1)
    measures: int = Field(ge=1)
    default_group_settings: GroupSettings
    group_settings_map: dict[int, GroupSettings] = {}

    @field_validator("time_signature", mode="before")
    @classmethod
    def _check_time_signature(cls, v: tuple[int, int] | Sequence[int]) -> TimeSignatureType:
        if len(v) != 2:
            raise ValueError(f"expected a 2-element sequence, got length {len(v)}")

        numerator, denominator = int(v[0]), int(v[1])
        validate_time_signature((numerator, denominator))

        return numerator, denominator

    def group_settings(self, group_id: int) -> GroupSettings:
        return self.group_settings_map.get(group_id, self.default_group_settings)
