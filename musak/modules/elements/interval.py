from typing import NamedTuple

from musak.modules.elements.constants import INTERVAL_NAMES
from musak.modules.elements.names import get_note_name


class Interval(NamedTuple):
    interval: int
    base_note_index: int = 0

    def get_base_note_name(self) -> str:
        if self.base_note_index:
            return get_note_name(self.base_note_index)

        return ""

    def __str__(self) -> str:
        base_note = self.get_base_note_name()
        return f"{base_note} {self.name} ({self.interval})"

    @property
    def name(self) -> str:
        return INTERVAL_NAMES[self.interval]

    @property
    def chord(self) -> list[int]:
        return [self.base_note_index, self.base_note_index + self.interval]
