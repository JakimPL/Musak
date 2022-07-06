from typing import NamedTuple

from modules.chords.auxiliary import get_note_name


class ChordInversion(NamedTuple):
    chord_type: str
    base_chord: tuple[int, ...]
    inversion_index: int
    base_note_index: int = 0

    def get_base_note_name(self) -> str:
        return get_note_name(self.chord[-self.inversion_index])

    def inversion_description(self) -> str:
        if self.inversion_index:
            return 'inversion no. {index}'.format(index=self.inversion_index)
        else:
            return 'root position'

    def __str__(self):
        return '{base_note}{chord_type}, {inversion_index}: {chord}'.format(
            chord=self.chord,
            base_note=self.get_base_note_name(),
            chord_type=self.chord_type,
            inversion_index=self.inversion_description()
        )

    @property
    def chord(self) -> list[int]:
        return [note + self.base_note_index for note in self.base_chord]
