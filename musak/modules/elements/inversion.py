from typing import NamedTuple

from musak.modules.elements.names import get_note_name


class ChordInversion(NamedTuple):
    chord_type: str
    base_chord: tuple[int, ...]
    inversion_index: int
    base_note_index: int = 0

    def get_base_note_name(self) -> str:
        return get_note_name(self.chord[-self.inversion_index])

    def inversion_description(self) -> str:
        if self.inversion_index:
            return f"inversion no. {self.inversion_index}"

        return "root position"

    def __str__(self) -> str:
        base_note = self.get_base_note_name()
        inversion_index = self.inversion_description()
        return f"{base_note} {self.chord_type}, {inversion_index}: {self.chord}"

    @property
    def chord(self) -> list[int]:
        return [note + self.base_note_index for note in self.base_chord]
