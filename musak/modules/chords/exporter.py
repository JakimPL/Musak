import abjad
from abjad.parsers.parser import LilyPondParser
from music21.base import Music21Object
from music21.chord import Chord
from music21.lily.translate import LilypondConverter
from music21.note import Note, Rest
from music21.stream.base import Stream
from music21.tempo import MetronomeMark

from musak.config.defaults import SEQUENTIAL, TEMPO
from musak.modules.elements.constants import (
    HALF_DURATION,
    QUARTER_DURATION,
    QUARTER_NOTE,
    WHOLE_DURATION,
)


def add_rest(
    stream: Stream[Music21Object],
    *,
    duration: str = HALF_DURATION,
) -> None:
    rest = Rest()
    rest.duration.type = duration
    stream.append(rest)  # type: ignore[no-untyped-call]


def create_sequence(
    iterable: list[int],
    stream: Stream[Music21Object],
    *,
    note_duration: str = QUARTER_DURATION,
) -> None:
    for midi_note in iterable:
        note = Note(midi_note)
        note.duration.type = note_duration
        stream.append(note)  # type: ignore[no-untyped-call]


def create_chord(
    iterable: list[int],
    stream: Stream[Music21Object],
    *,
    duration: str = WHOLE_DURATION,
) -> None:
    chord = Chord(iterable)
    chord.duration.type = duration
    stream.append(chord)  # type: ignore[no-untyped-call]


def create_stream(
    iterable: list[int],
    *,
    tempo: int = TEMPO,
    sequential: bool = SEQUENTIAL,
) -> Stream[Music21Object]:
    stream = Stream[Music21Object]()
    mark = MetronomeMark(number=tempo)
    stream.append(mark)  # type: ignore[no-untyped-call]
    if sequential:
        create_sequence(iterable, stream)
        add_rest(stream)
    else:
        create_chord(iterable, stream)
        add_rest(stream, duration=WHOLE_DURATION)

    return stream


def to_abjad(
    iterable: list[int],
    *,
    tempo: int = TEMPO,
    sequential: bool = SEQUENTIAL,
) -> abjad.Score:
    stream = create_stream(
        iterable,
        tempo=tempo,
        sequential=sequential,
    )

    ly_converter = LilypondConverter()  # type: ignore[no-untyped-call]
    ly_stream = ly_converter.lySequentialMusicFromStream(stream)  # type: ignore[no-untyped-call]

    parser = LilyPondParser("nederlands")  # type: ignore[no-untyped-call]
    staff = parser(str(ly_stream))

    abjad_tempo = abjad.MetronomeMark(abjad.Duration(*QUARTER_NOTE), tempo)
    abjad.attach(abjad_tempo, staff[0])
    score = abjad.Score([staff])
    return score
