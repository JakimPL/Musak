import abjad

from musak.modules.elements.constants import DEFAULT_TEMPO, QUARTER_NOTE
from musak.modules.elements.exceptions import EmptyScoreException
from musak.modules.elements.phrase import Phrase
from musak.modules.elements.time_signature import (
    DEFAULT_TIME_SIGNATURE,
    TimeSignatureType,
)
from musak.modules.rhythm.exceptions import InvalidBeatException


def to_abjad_string(
    phrase: Phrase,
    *,
    time_signature: TimeSignatureType = DEFAULT_TIME_SIGNATURE,
) -> str:
    invalid_beat = phrase.find_invalid_beat(time_signature=time_signature)
    if invalid_beat:
        raise InvalidBeatException(f"invalid beat no. {invalid_beat}")

    return " ".join(str(note) for note in phrase.notes)


def to_abjad_score(
    phrases: list[Phrase],
    *,
    time_signature: TimeSignatureType = DEFAULT_TIME_SIGNATURE,
    tempo: int = DEFAULT_TEMPO,
) -> abjad.Score:
    if not phrases:
        raise EmptyScoreException("an empty score")

    abjad_signature = abjad.TimeSignature(time_signature)
    abjad_tempo = abjad.MetronomeMark(abjad.Duration(*QUARTER_NOTE), tempo)

    staves = []
    for notes in phrases:
        voice = abjad.Voice(
            to_abjad_string(notes, time_signature=time_signature),
            name="Rhythm",
        )
        abjad.attach(abjad_tempo, voice[0])
        abjad.attach(abjad_signature, voice[0])
        staff = abjad.Staff([voice], lilypond_type="RhythmicStaff", name="Percussion")
        staff_group = abjad.StaffGroup([staff])
        staves.append(staff_group)

    abjad_score = abjad.Score(staves)
    return abjad_score
