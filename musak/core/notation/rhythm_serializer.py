from __future__ import annotations

from fractions import Fraction
from typing import Final

from musak.modules.elements.note import Note
from musak.modules.elements.phrase import Phrase
from musak_shared.notation.schema import (
    EIGHTH,
    HALF,
    QUARTER,
    REST_SUFFIX,
    SIXTEENTH,
    THIRTY_SECOND,
    WHOLE,
    Clef,
    NoteData,
    ScoreData,
    StaveData,
    VexflowDuration,
    VoiceData,
)
from musak_shared.time_signature import TimeSignatureType

PERCUSSION_CLEF: Final[Clef] = "percussion"
PERCUSSION_KEY: Final[str] = "b/4"
TREBLE_CLEF: Final[Clef] = "treble"
_MELODIC_NOTE_PATTERN: Final[list[str]] = [
    "c/4",
    "e/4",
    "g/4",
    "b/3",
]

_DENOMINATOR_TO_DURATION: Final[dict[int, str]] = {
    1: WHOLE,
    2: HALF,
    4: QUARTER,
    8: EIGHTH,
    16: SIXTEENTH,
    32: THIRTY_SECOND,
}


def _get_melodic_key_for_group(group_index: int) -> str:
    pattern_index = group_index % len(_MELODIC_NOTE_PATTERN)
    octave_offset = group_index // len(_MELODIC_NOTE_PATTERN)

    key = _MELODIC_NOTE_PATTERN[pattern_index]
    pitch, octave_str = key.split("/")
    octave = int(octave_str) + octave_offset
    return f"{pitch}/{octave}"


def note_to_note_data(note: Note, melodic_key: str | None = None) -> NoteData:
    _, denominator, dot_count = note.dots
    base = _DENOMINATOR_TO_DURATION[denominator]
    duration: VexflowDuration = base + (REST_SUFFIX if note.pause else "")  # type: ignore[assignment]
    key = melodic_key if melodic_key else PERCUSSION_KEY
    return NoteData(keys=[key], duration=duration, dots=dot_count)


def phrase_to_voice_data(phrase: Phrase, melodic_key: str | None = None) -> VoiceData:
    return VoiceData(notes=[note_to_note_data(note, melodic_key) for note in phrase.notes])


def split_into_measures(phrase: Phrase, time_signature: TimeSignatureType) -> list[Phrase]:
    measure_length = Fraction(*time_signature)
    measures: list[Phrase] = []
    current_notes: list[Note] = []
    accumulated = Fraction(0)

    for note in phrase.notes:
        current_notes.append(note)
        accumulated += note.duration
        if accumulated == measure_length:
            measures.append(Phrase(notes=current_notes))
            current_notes = []
            accumulated = Fraction(0)

    if current_notes:
        measures.append(Phrase(notes=current_notes))

    return measures


def _measure_to_stave_data(
    measure: Phrase,
    time_signature: TimeSignatureType,
    show_time_signature: bool,
    clef: Clef = PERCUSSION_CLEF,
    melodic_key: str | None = None,
) -> StaveData:
    voice = phrase_to_voice_data(measure, melodic_key)
    return StaveData(
        clef=clef,
        time_signature=time_signature if show_time_signature else None,
        voices=[voice],
    )


def phrases_to_score_data(
    phrases: list[Phrase],
    *,
    time_signature: TimeSignatureType,
    tempo: int,
    max_notes_per_measure: int | None = None,
    melodic: bool = False,
) -> ScoreData:
    rows: list[list[StaveData]] = []

    clef = TREBLE_CLEF if melodic else PERCUSSION_CLEF
    for group_index, phrase in enumerate(phrases):
        measures = split_into_measures(phrase, time_signature)
        melodic_key = _get_melodic_key_for_group(group_index) if melodic else None
        row = [
            _measure_to_stave_data(measure, time_signature, measure_index == 0, clef, melodic_key)
            for measure_index, measure in enumerate(measures)
        ]
        rows.append(row)

    return ScoreData(
        rows=rows,
        tempo=tempo,
        max_notes_per_measure=max_notes_per_measure,
    )
