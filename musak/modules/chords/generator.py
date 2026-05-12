import random
from collections import OrderedDict
from collections.abc import Sequence

from musak.config.defaults import HIGHEST_NOTE, LOWEST_NOTE
from musak.modules.elements.constants import CHORDS
from musak.modules.elements.interval import Interval
from musak.modules.elements.inversion import ChordInversion


def generate_chord_inversions(chord: Sequence[int]) -> list[tuple[int, ...]]:
    base_chord = list(chord)
    inversions = []
    for index, note in enumerate(chord):
        new_chord = []
        for base_index in range(len(chord)):
            new_note = base_chord[base_index] - note
            if base_index < index:
                new_note += 12

            new_chord.append(new_note)

        inversion = tuple(sorted(new_chord))
        inversions.append(inversion)

    return list(OrderedDict.fromkeys(inversions))


def get_random_chord_type(
    chords: dict[str, list[ChordInversion]] | None = None,
) -> str:
    candidates = chords if chords else CHORDS
    return random.choice(list(candidates.keys()))


def get_random_inversion(chord: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    inversions = generate_chord_inversions(chord)
    index = random.randrange(0, len(inversions))
    inversion = inversions[index]
    return index, inversion


def get_random_base_note(
    chord: Sequence[int],
    *,
    lowest_note: int = LOWEST_NOTE,
    highest_note: int = HIGHEST_NOTE,
) -> int:
    if lowest_note > highest_note:
        lowest_note, highest_note = highest_note, lowest_note

    min_note = min(chord)
    normalized_chord = [note - min_note for note in chord]
    chord_range = max(normalized_chord)

    real_highest_note = max(highest_note - chord_range, lowest_note + chord_range)
    base = random.randint(lowest_note, real_highest_note)

    return base


def get_random_chord_inversion(
    inversions: dict[str, list[ChordInversion]],
    *,
    lowest_note: int = LOWEST_NOTE,
    highest_note: int = HIGHEST_NOTE,
) -> ChordInversion:
    chord_type = get_random_chord_type(inversions)
    inversion_index = random.randrange(len(inversions[chord_type]))
    inversion = inversions[chord_type][inversion_index]
    chord = inversion.base_chord
    base_note_index = get_random_base_note(
        chord,
        lowest_note=lowest_note,
        highest_note=highest_note,
    )

    return ChordInversion(
        chord_type=chord_type,
        base_chord=chord,
        inversion_index=inversion_index,
        base_note_index=base_note_index,
    )


def get_random_interval(
    intervals: dict[str, int],
    *,
    lowest_note: int = LOWEST_NOTE,
    highest_note: int = HIGHEST_NOTE,
) -> Interval:
    interval_value = random.choice(list(intervals.values()))
    base_note_index = get_random_base_note(
        (0, interval_value),
        lowest_note=lowest_note,
        highest_note=highest_note,
    )

    return Interval(interval=interval_value, base_note_index=base_note_index)


def generate_all_inversions(
    chords: dict[str, Sequence[int]] | None = None,
    *,
    reduce: bool = True,
) -> dict[str, list[ChordInversion]]:
    candidates = chords if chords else CHORDS
    all_inversions = {chord_name: generate_chord_inversions(chord) for chord_name, chord in candidates.items()}

    if not reduce:
        all_chords = all_inversions
    else:
        all_chords = {}
        for chord_type in all_inversions:
            if all_inversions[chord_type] and all_inversions[chord_type][0] not in sum(all_chords.values(), []):
                all_chords[chord_type] = all_inversions[chord_type]

    return {
        chord_type: [
            ChordInversion(
                chord_type=chord_type,
                base_chord=chord,
                inversion_index=index,
            )
            for index, chord in enumerate(inversions)
        ]
        for chord_type, inversions in all_chords.items()
    }
