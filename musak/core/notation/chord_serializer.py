from __future__ import annotations

from typing import Final, cast

from musak.core.notation.schema import (
    HALF,
    QUARTER,
    REST_SUFFIX,
    WHOLE,
    Clef,
    NoteData,
    ScoreData,
    StaveData,
    VexflowDuration,
    VoiceData,
)
from musak.modules.elements.interval import Interval
from musak.modules.elements.inversion import ChordInversion
from musak.modules.elements.names import midi_to_vexflow_key

MIDDLE_C: Final[int] = 60
WIDE_SPAN_THRESHOLD: Final[int] = 24


def _select_clef(midi_notes: list[int]) -> Clef:
    span = max(midi_notes) - min(midi_notes)
    if span >= WIDE_SPAN_THRESHOLD:
        return "treble"

    if max(midi_notes) < MIDDLE_C:
        return "bass"

    return "treble"


def _whole_rest() -> NoteData:
    return NoteData(keys=[], duration=cast(VexflowDuration, WHOLE + REST_SUFFIX))


def _half_rest() -> NoteData:
    return NoteData(keys=[], duration=cast(VexflowDuration, HALF + REST_SUFFIX))


def _chord_voice(midi_notes: list[int]) -> VoiceData:
    keys = [midi_to_vexflow_key(midi_note) for midi_note in midi_notes]
    chord_note = NoteData(keys=keys, duration=WHOLE)
    return VoiceData(notes=[chord_note, _whole_rest()])


def _sequential_voice(midi_notes: list[int]) -> VoiceData:
    quarter_notes = [NoteData(keys=[midi_to_vexflow_key(midi_note)], duration=QUARTER) for midi_note in midi_notes]

    return VoiceData(notes=quarter_notes + [_half_rest()])


def interval_to_score_data(
    interval: Interval,
    *,
    sequential: bool,
    tempo: int,
) -> ScoreData:
    midi_notes = interval.chord
    clef = _select_clef(midi_notes)
    voice = _sequential_voice(midi_notes) if sequential else _chord_voice(midi_notes)
    stave = StaveData(clef=clef, voices=[voice])
    return ScoreData(staves=[stave], tempo=tempo)


def inversion_to_score_data(
    inversion: ChordInversion,
    *,
    tempo: int,
) -> ScoreData:
    midi_notes = inversion.chord
    clef = _select_clef(midi_notes)
    voice = _chord_voice(midi_notes)
    stave = StaveData(clef=clef, voices=[voice])
    return ScoreData(staves=[stave], tempo=tempo)
