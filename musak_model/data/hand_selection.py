from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Final

from music21 import chord, instrument, note
from music21.stream.base import Part, Score

_EXPECTED_PIANO_PARTS: Final[int] = 2
_PIANO_MIDI_PROGRAMS: Final[frozenset[int]] = frozenset({0})
_PITCH_CENTER_DECIMALS: Final[int] = 0


@dataclass(frozen=True)
class HandParts:
    right: Part
    left: Part


class PianoHandSelectionError(ValueError):
    pass


class InvalidPianoPartCountError(PianoHandSelectionError):
    pass


class NonPianoInstrumentError(PianoHandSelectionError):
    pass


class EmptyPitchedPartError(PianoHandSelectionError):
    pass


class AmbiguousHandAssignmentError(PianoHandSelectionError):
    pass


def select_piano_hand_parts(score: Score) -> HandParts:
    parts = [part for part in score.parts if isinstance(part, Part)]
    if len(parts) != _EXPECTED_PIANO_PARTS:
        raise InvalidPianoPartCountError(f"expected exactly 2 score parts, found {len(parts)}")

    for index, part in enumerate(parts):
        if not _is_allowed_piano_part(part):
            raise NonPianoInstrumentError(f"part {index} has an explicit non-piano instrument")

    pitch_centers = [_pitch_center(part) for part in parts]
    rounded_pitch_centers = [round(center, _PITCH_CENTER_DECIMALS) for center in pitch_centers]
    if rounded_pitch_centers[0] == rounded_pitch_centers[1]:
        raise AmbiguousHandAssignmentError("cannot assign hands: piano parts have identical pitch centers")

    right_index = 0 if pitch_centers[0] > pitch_centers[1] else 1
    left_index = 1 - right_index
    return HandParts(right=parts[right_index], left=parts[left_index])


def _is_allowed_piano_part(part: Part) -> bool:
    part_instrument = part.getInstrument(returnDefault=False)
    if part_instrument is None:
        return True

    if isinstance(part_instrument, instrument.Piano):
        return True

    midi_program = getattr(part_instrument, "midiProgram", None)
    return isinstance(midi_program, int) and midi_program in _PIANO_MIDI_PROGRAMS


def _pitch_center(part: Part) -> float:
    pitches = _part_midi_pitches(part)
    if not pitches:
        raise EmptyPitchedPartError("cannot assign hands: piano part has no pitched events")

    return float(median(pitches))


def _part_midi_pitches(part: Part) -> list[int]:
    pitches: list[int] = []
    for element in part.flatten().notes:
        if isinstance(element, note.Note):
            pitches.append(element.pitch.midi)
        elif isinstance(element, chord.Chord):
            pitches.extend(pitch.midi for pitch in element.pitches)

    return pitches
