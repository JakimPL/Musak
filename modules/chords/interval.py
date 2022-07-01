from typing import NamedTuple

from modules.chords.auxiliary import get_note_name
from modules.chords.constants import INTERVAL_NAMES


class Interval(NamedTuple):
    interval: int
    base_note_index: int = 0

    def get_base_note_name(self) -> str:
        if self.base_note_index:
            return get_note_name(self.base_note_index)
        else:
            return ''

    def __str__(self):
        return '{base_note} {name} ({interval})'.format(
            interval=self.interval,
            base_note=self.get_base_note_name(),
            name=self.name
        )

    @property
    def name(self) -> str:
        return INTERVAL_NAMES[self.interval]

    @property
    def chord(self) -> list[int]:
        return [self.base_note_index, self.base_note_index + self.interval]
