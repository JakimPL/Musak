from __future__ import annotations

from fractions import Fraction
from typing import Final

from musak.core.notation.schema import (
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
from musak.modules.elements.note import Note
from musak.modules.elements.phrase import Phrase
from musak.modules.elements.time_signature import TimeSignatureType

PERCUSSION_CLEF: Final[Clef] = "percussion"
PERCUSSION_KEY: Final[str] = "b/4"

_DENOMINATOR_TO_DURATION: Final[dict[int, str]] = {
    1: WHOLE,
    2: HALF,
    4: QUARTER,
    8: EIGHTH,
    16: SIXTEENTH,
    32: THIRTY_SECOND,
}


def note_to_note_data(note: Note) -> NoteData:
    _, denominator, dot_count = note.dots
    base = _DENOMINATOR_TO_DURATION[denominator]
    duration: VexflowDuration = base + (REST_SUFFIX if note.pause else "")  # type: ignore[assignment]
    return NoteData(keys=[PERCUSSION_KEY], duration=duration, dots=dot_count)


def phrase_to_voice_data(phrase: Phrase) -> VoiceData:
    return VoiceData(notes=[note_to_note_data(note) for note in phrase.notes])


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
) -> StaveData:
    voice = phrase_to_voice_data(measure)
    return StaveData(
        clef=PERCUSSION_CLEF,
        time_signature=time_signature if show_time_signature else None,
        voices=[voice],
    )


def phrases_to_score_data(
    phrases: list[Phrase],
    *,
    time_signature: TimeSignatureType,
    tempo: int,
) -> ScoreData:
    rows: list[list[StaveData]] = []
    for phrase in phrases:
        measures = split_into_measures(phrase, time_signature)
        row = [
            _measure_to_stave_data(measure, time_signature, measure_index == 0)
            for measure_index, measure in enumerate(measures)
        ]
        rows.append(row)
    return ScoreData(rows=rows, tempo=tempo)
