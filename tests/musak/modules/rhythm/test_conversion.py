from typing import NamedTuple

import mido
import pytest

from musak.modules.elements.constants import (
    MIDI_MELODIC_CHANNEL,
    MIDI_MELODIC_NOTE,
    MIDI_PERCUSSION_CHANNEL,
)
from musak.modules.elements.note import Note
from musak.modules.elements.phrase import Phrase
from musak.modules.rhythm.conversion import phrases_to_midi
from musak.modules.rhythm.exceptions import InvalidBeatException


class MidiEventCase(NamedTuple):
    phrases: list[Phrase]
    melodic: bool
    expected_messages: list[tuple[str, int, int, int, int]]


def _extract_message_tuples(track: list[mido.Message]) -> list[tuple[str, int, int, int, int]]:
    absolute_time = 0
    events: list[tuple[str, int, int, int, int]] = []

    for message in track:
        if message.type not in ("note_on", "note_off"):
            continue

        absolute_time += message.time
        events.append((message.type, message.channel, message.note, absolute_time, message.velocity))

    return events


MELODY_CASES = [
    MidiEventCase(
        phrases=[Phrase(notes=[Note(duration=1)])],
        melodic=True,
        expected_messages=[
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 0, 80),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 1920, 0),
        ],
    ),
    MidiEventCase(
        phrases=[Phrase(notes=[Note(duration=4), Note(duration=4), Note(duration=2)])],
        melodic=True,
        expected_messages=[
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 0, 80),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 480, 0),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 480, 80),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 960, 0),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 960, 80),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 1920, 0),
        ],
    ),
    MidiEventCase(
        phrases=[Phrase(notes=[Note(duration=2), Note(duration=4), Note(duration=4)])],
        melodic=True,
        expected_messages=[
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 0, 80),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 960, 0),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 960, 80),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 1440, 0),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 1440, 80),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 1920, 0),
        ],
    ),
    MidiEventCase(
        phrases=[
            Phrase(notes=[Note(duration=2), Note(duration=4), Note(duration=4)]),
            Phrase(notes=[Note(duration=4), Note(duration=2), Note(duration=4)]),
            Phrase(notes=[Note(duration=4), Note(duration=4), Note(duration=2)]),
            Phrase(notes=[Note(duration=1)]),
            Phrase(notes=[Note(duration=1)]),
        ],
        melodic=True,
        expected_messages=[
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 0, 80),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 4, 0, 80),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 7, 0, 80),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 10, 0, 80),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 12, 0, 80),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 4, 480, 0),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 7, 480, 0),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 4, 480, 80),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 7, 480, 80),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 960, 0),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 7, 960, 0),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 960, 80),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 7, 960, 80),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 1440, 0),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 4, 1440, 0),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 1440, 80),
            ("note_on", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 4, 1440, 80),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE, 1920, 0),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 4, 1920, 0),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 7, 1920, 0),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 10, 1920, 0),
            ("note_off", MIDI_MELODIC_CHANNEL, MIDI_MELODIC_NOTE + 12, 1920, 0),
        ],
    ),
]


PERCUSSION_CASES = [
    MidiEventCase(
        phrases=[Phrase(notes=[Note(duration=1)])],
        melodic=False,
        expected_messages=[
            ("note_on", MIDI_PERCUSSION_CHANNEL, 36, 0, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 36, 1920, 0),
        ],
    ),
    MidiEventCase(
        phrases=[Phrase(notes=[Note(duration=4), Note(duration=4), Note(duration=2)])],
        melodic=False,
        expected_messages=[
            ("note_on", MIDI_PERCUSSION_CHANNEL, 36, 0, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 36, 480, 0),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 36, 480, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 36, 960, 0),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 36, 960, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 36, 1920, 0),
        ],
    ),
    MidiEventCase(
        phrases=[Phrase(notes=[Note(duration=2), Note(duration=4), Note(duration=4)])],
        melodic=False,
        expected_messages=[
            ("note_on", MIDI_PERCUSSION_CHANNEL, 36, 0, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 36, 960, 0),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 36, 960, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 36, 1440, 0),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 36, 1440, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 36, 1920, 0),
        ],
    ),
    MidiEventCase(
        phrases=[
            Phrase(notes=[Note(duration=-4), Note(duration=4), Note(duration=-2)]),
            Phrase(notes=[Note(duration=-2), Note(duration=2)]),
        ],
        melodic=False,
        expected_messages=[
            ("note_on", MIDI_PERCUSSION_CHANNEL, 36, 480, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 36, 960, 0),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 42, 960, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 42, 1920, 0),
        ],
    ),
    MidiEventCase(
        phrases=[
            Phrase(notes=[Note(duration=2), Note(duration=4), Note(duration=4)]),
            Phrase(notes=[Note(duration=4), Note(duration=2), Note(duration=4)]),
            Phrase(notes=[Note(duration=4), Note(duration=4), Note(duration=2)]),
            Phrase(notes=[Note(duration=1)]),
            Phrase(notes=[Note(duration=1)]),
        ],
        melodic=False,
        expected_messages=[
            ("note_on", MIDI_PERCUSSION_CHANNEL, 36, 0, 80),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 36, 0, 80),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 37, 0, 80),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 38, 0, 80),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 42, 0, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 38, 480, 0),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 42, 480, 0),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 38, 480, 80),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 42, 480, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 36, 960, 0),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 38, 960, 0),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 36, 960, 80),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 38, 960, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 36, 1440, 0),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 42, 1440, 0),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 36, 1440, 80),
            ("note_on", MIDI_PERCUSSION_CHANNEL, 42, 1440, 80),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 36, 1920, 0),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 36, 1920, 0),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 37, 1920, 0),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 38, 1920, 0),
            ("note_off", MIDI_PERCUSSION_CHANNEL, 42, 1920, 0),
        ],
    ),
]


@pytest.mark.parametrize("case", MELODY_CASES)
def test_phrases_to_midi_melodic(case: MidiEventCase) -> None:
    midi = phrases_to_midi(case.phrases, melodic=case.melodic)
    track = midi.tracks[0]
    assert _extract_message_tuples(track) == case.expected_messages


@pytest.mark.parametrize("case", PERCUSSION_CASES)
def test_phrases_to_midi_percussion(case: MidiEventCase) -> None:
    midi = phrases_to_midi(case.phrases, melodic=case.melodic)
    track = midi.tracks[0]
    assert _extract_message_tuples(track) == case.expected_messages


def test_phrases_to_midi_invalid_beat_exception_message() -> None:
    phrase = Phrase(notes=[Note(duration=4), Note(duration=4), Note(duration=4)])

    with pytest.raises(InvalidBeatException) as exc_info:
        phrases_to_midi([phrase], melodic=False)

    message = str(exc_info.value)
    assert "invalid beat no. 1" in message
    assert "length 3/4" in message
    assert "required boundary 1" in message
