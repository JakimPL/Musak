import abjad

from modules.rhythm.constants import DEFAULT_TEMPO, QUARTER_NOTE
from modules.rhythm.exceptions import EmptyScoreException, InvalidBeatException
from modules.rhythm.phrase import Phrase
from modules.rhythm.time_signature import DEFAULT_TIME_SIGNATURE, TimeSignatureType


def to_abjad_string(
    phrase: Phrase,
    *,
    time_signature: TimeSignatureType = DEFAULT_TIME_SIGNATURE,
) -> str:
    invalid_beat = phrase.find_invalid_beat(time_signature=time_signature)
    if invalid_beat:
        raise InvalidBeatException("invalid beat no. {beat}".format(beat=invalid_beat))

    abjad_string = ""
    for note in phrase.notes:
        abjad_string += "{note} ".format(note=note)

    return abjad_string


def to_abjad_score(
    phrases: list[Phrase],
    *,
    time_signature: TimeSignatureType = DEFAULT_TIME_SIGNATURE,
    tempo: int = DEFAULT_TEMPO,
) -> abjad.Score:
    if not phrases:
        raise EmptyScoreException("an empty score")

    abjad_signature = abjad.TimeSignature(time_signature)
    abjad_tempo = abjad.MetronomeMark(QUARTER_NOTE, tempo)  # type: ignore[arg-type]

    staves = []
    for notes in phrases:
        voice = abjad.Voice(
            to_abjad_string(notes, time_signature=time_signature), name="Rhythm"
        )
        abjad.attach(abjad_tempo, voice[0])
        abjad.attach(abjad_signature, voice[0])
        staff = abjad.Staff([voice], lilypond_type="RhythmicStaff", name="Percussion")
        staff_group = abjad.StaffGroup([staff])
        staves.append(staff_group)

    abjad_score = abjad.Score(staves)
    return abjad_score
