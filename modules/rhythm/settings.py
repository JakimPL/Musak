from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from modules.rhythm.misc import check_type, is_power_of_two
from modules.rhythm.note import Note, NoteType
from modules.rhythm.phrase import Phrase, PhraseType
from modules.rhythm.time_signature import TimeSignatureType


class GroupSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    notes: list[Note] = []
    phrases: list[Phrase] = []

    @field_validator("notes", mode="before")
    @classmethod
    def _coerce_notes(cls, v: object) -> list[Note]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [n if isinstance(n, Note) else Note(n) for n in v]  # type: ignore[arg-type]
        raise TypeError(f"expected a list, got {type(v)}")

    @field_validator("phrases", mode="before")
    @classmethod
    def _coerce_phrases(cls, v: object) -> list[Phrase]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [p if isinstance(p, Phrase) else Phrase(list(p)) for p in v]  # type: ignore[arg-type]
        raise TypeError(f"expected a list, got {type(v)}")

    @model_validator(mode="after")
    def _validate_non_empty(self) -> "GroupSettings":
        if not self.get_all_phrases():
            raise ValueError("notes and phrases cannot be both empty")
        return self

    def get_all_phrases(self) -> list[Phrase]:
        return [Phrase([note]) for note in self.notes] + self.phrases


class Settings:
    def __init__(self) -> None:
        self._time_signature: TimeSignatureType = (4, 4)
        self._tempo: int = 120
        self._groups: int = 2
        self._measures: int = 2
        self._group_settings: Dict[int, GroupSettings] = {}
        self._default_group_settings: GroupSettings = GroupSettings(
            notes=[-8, -4, -2, 2, 4, 8],
            phrases=[[4, -4], [-4, 4], [8, 8, 8, 8], [4, 4]],
        )

    def group_settings(self, group_id: int) -> GroupSettings:
        return (
            self._group_settings[group_id]
            if group_id in self._group_settings
            else self._default_group_settings
        )

    def set_group(self, group_id: int, group_settings: GroupSettings):
        check_type(group_settings, GroupSettings)
        self._group_settings[group_id] = group_settings

    @property
    def time_signature(self) -> TimeSignatureType:
        return self._time_signature

    @time_signature.setter
    def time_signature(self, signature: TimeSignatureType):
        if not (
            isinstance(signature, tuple)
            and len(signature) == 2
            and all([isinstance(element, int) for element in signature])
        ):
            raise TypeError(
                "expected a 2-tuple (int, int), got {type}".format(type=type(signature))
            )

        if not all([element > 0 for element in signature]):
            raise ValueError(
                "non-positive element in a signature: {}".format(signature)
            )

        if not is_power_of_two(signature[1]):
            raise ValueError("time signature denominator has to be a power of two")

        self._time_signature = signature

    @property
    def groups(self) -> int:
        return self._groups

    @groups.setter
    def groups(self, groups: int):
        check_type(groups, int)

        if groups <= 0:
            raise ValueError(
                "number of groups cannot be non-positive, got {value}".format(
                    value=groups
                )
            )

        self._groups = groups

    @property
    def measures(self) -> int:
        return self._measures

    @measures.setter
    def measures(self, measures: int):
        check_type(measures, int)

        if measures <= 0:
            raise ValueError(
                "number of measures cannot be non-positive, got {value}".format(
                    value=measures
                )
            )

        self._measures = measures

    @property
    def tempo(self) -> int:
        return self._tempo

    @tempo.setter
    def tempo(self, tempo: int):
        check_type(tempo, int)

        if tempo <= 0:
            raise ValueError(
                "number of groups cannot be non-positive, got {value}".format(
                    value=tempo
                )
            )

        self._tempo = tempo

    @property
    def default_group_settings(self) -> GroupSettings:
        return self._default_group_settings

    @default_group_settings.setter
    def default_group_settings(self, group_settings: GroupSettings):
        if not isinstance(group_settings, GroupSettings):
            raise TypeError(
                "expected GroupSettings object, got {type}".format(
                    type=type(group_settings)
                )
            )

        self._default_group_settings = group_settings
