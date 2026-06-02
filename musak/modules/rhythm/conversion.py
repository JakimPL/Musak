import pathlib
from fractions import Fraction
from typing import Callable, NamedTuple

import mido

from musak.config.defaults import MELODIC, TEMPO, TIME_SIGNATURE
from musak.modules.elements.constants import (
    MIDI_MELODIC_CHANNEL,
    MIDI_MELODIC_NOTE,
    MIDI_PERCUSSION_CHANNEL,
    MIDI_PERCUSSION_NOTES,
    MIDI_TICKS_PER_BEAT,
    MIDI_VELOCITY,
)
from musak.modules.elements.exceptions import EmptyScoreException
from musak.modules.elements.phrase import Phrase
from musak.modules.rhythm.exceptions import InvalidBeatException
from musak_shared.time_signature import TimeSignatureType


def _tempo_to_us(tempo: int) -> int:
    return 60_000_000 // tempo


def _note_ticks(duration: Fraction) -> int:
    return round(float(duration) * 4 * MIDI_TICKS_PER_BEAT)


class _MidiEvent(NamedTuple):
    time: Fraction
    channel: int
    note: int
    is_on: bool
    velocity: int


def _get_melodic_note_for_group(group_index: int) -> int:
    pattern = [0, 4, 7, 10]
    octave_index = group_index // 4
    note_in_pattern = group_index % 4
    return MIDI_MELODIC_NOTE + (octave_index * 12) + pattern[note_in_pattern]


def phrases_to_midi(
    phrases: list[Phrase],
    *,
    time_signature: TimeSignatureType = TIME_SIGNATURE,
    tempo: int = TEMPO,
    melodic: bool = MELODIC,
) -> mido.MidiFile:
    if not phrases:
        raise EmptyScoreException("an empty score")

    num, den = time_signature

    for phrase_index, phrase in enumerate(phrases):
        invalid_beat = phrase.find_invalid_beat(time_signature=time_signature)
        if invalid_beat:
            boundary = invalid_beat * Fraction(num, den)
            raise InvalidBeatException(
                f"invalid beat no. {invalid_beat}: phrase {phrase_index} length {phrase.length} "
                f"does not add up to required boundary {boundary} for time signature {num}/{den}"
            )

    mid = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    num, den = time_signature
    track.append(mido.MetaMessage("set_tempo", tempo=_tempo_to_us(tempo), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0))

    events = _build_events(phrases, melodic)
    _append_events(track, events)

    return mid


def _build_events(phrases: list[Phrase], melodic: bool) -> list[_MidiEvent]:
    if melodic:
        return _build_melodic_events(phrases)

    return _build_percussion_events(phrases)


def _build_melodic_events(phrases: list[Phrase]) -> list[_MidiEvent]:
    return _collect_events(
        phrases,
        _get_melodic_note_for_group,
        MIDI_MELODIC_CHANNEL,
    )


def _build_percussion_events(phrases: list[Phrase]) -> list[_MidiEvent]:
    return _collect_events(
        phrases,
        lambda group_index: MIDI_PERCUSSION_NOTES[group_index % len(MIDI_PERCUSSION_NOTES)],
        MIDI_PERCUSSION_CHANNEL,
    )


def _collect_events(
    phrases: list[Phrase],
    note_value_for_group: Callable[[int], int],
    channel: int,
) -> list[_MidiEvent]:
    events: list[_MidiEvent] = []

    for group_index, phrase in enumerate(phrases):
        note_value = note_value_for_group(group_index)
        time_position = Fraction(0)

        for note in phrase.notes:
            if note.pause:
                time_position += note.duration
                continue

            events.append(_MidiEvent(time_position, channel, note_value, True, MIDI_VELOCITY))
            time_position += note.duration
            events.append(_MidiEvent(time_position, channel, note_value, False, 0))

    return sorted(events, key=_event_sort_key)


def _event_sort_key(event: _MidiEvent) -> tuple[Fraction, int, bool, int]:
    return event.time, event.channel, event.is_on, event.note


def _append_events(track: mido.MidiTrack, events: list[_MidiEvent]) -> None:
    last_time = Fraction(0)

    for event in events:
        delta = event.time - last_time
        track.append(
            mido.Message(
                "note_on" if event.is_on else "note_off",
                channel=event.channel,
                note=event.note,
                velocity=event.velocity,
                time=_note_ticks(delta),
            )
        )
        last_time = event.time


def save_midi(
    phrases: list[Phrase],
    path: pathlib.Path,
    *,
    time_signature: TimeSignatureType = TIME_SIGNATURE,
    tempo: int = TEMPO,
    melodic: bool = MELODIC,
) -> pathlib.Path:
    mid = phrases_to_midi(phrases, time_signature=time_signature, tempo=tempo, melodic=melodic)
    mid.save(str(path))
    return path
