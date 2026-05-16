import io
from collections.abc import Iterable
from fractions import Fraction
from typing import Final

import mido

from musak_model.decoder import PianoRollEvent
from musak_model.tokens.schema import Hand
from musak_shared.exporter import Exporter

_QUARTERS_PER_WHOLE: Final[int] = 4
_DEFAULT_TICKS_PER_BEAT: Final[int] = 480
_PIANO_PROGRAM: Final[int] = 0
_DEFAULT_VELOCITY: Final[int] = 72
_HAND_CHANNELS: Final[dict[Hand, int]] = {
    Hand.RIGHT: 0,
    Hand.LEFT: 1,
}


def piano_roll_events_to_midi_file(
    events: Iterable[PianoRollEvent],
    *,
    bpm: int,
    hands: frozenset[Hand] = frozenset(Hand),
    ticks_per_beat: int = _DEFAULT_TICKS_PER_BEAT,
) -> mido.MidiFile:
    midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    tempo_track.append(mido.MetaMessage("end_of_track", time=0))
    midi_file.tracks.append(tempo_track)

    for hand in Hand:
        hand_events = [event for event in events if event.hand == hand and event.hand in hands]
        if hand_events:
            midi_file.tracks.append(_events_to_track(hand_events, hand=hand, ticks_per_beat=ticks_per_beat))

    return midi_file


def piano_roll_events_to_audio_data(
    events: Iterable[PianoRollEvent],
    *,
    bpm: int,
    hands: frozenset[Hand] = frozenset(Hand),
    exporter: Exporter | None = None,
) -> str:
    midi_file = piano_roll_events_to_midi_file(events, bpm=bpm, hands=hands)
    renderer = Exporter() if exporter is None else exporter
    return renderer.export_audio(midi_file_bytes(midi_file))


def midi_file_bytes(midi_file: mido.MidiFile) -> bytes:
    buffer = io.BytesIO()
    midi_file.save(file=buffer)
    return buffer.getvalue()


def _events_to_track(events: list[PianoRollEvent], *, hand: Hand, ticks_per_beat: int) -> mido.MidiTrack:
    channel = _HAND_CHANNELS[hand]
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", channel=channel, program=_PIANO_PROGRAM, time=0))
    timed_messages: list[tuple[int, int, mido.Message]] = []

    for event in events:
        start_tick = _whole_note_fraction_to_ticks(event.start, ticks_per_beat=ticks_per_beat)
        end_tick = _whole_note_fraction_to_ticks(event.end, ticks_per_beat=ticks_per_beat)
        timed_messages.append(
            (
                start_tick,
                1,
                mido.Message(
                    "note_on",
                    channel=channel,
                    note=event.midi_pitch,
                    velocity=_DEFAULT_VELOCITY,
                    time=0,
                ),
            )
        )
        timed_messages.append(
            (
                end_tick,
                0,
                mido.Message("note_off", channel=channel, note=event.midi_pitch, velocity=0, time=0),
            )
        )

    previous_tick = 0
    for absolute_tick, _, message in sorted(timed_messages, key=lambda item: (item[0], item[1])):
        message.time = absolute_tick - previous_tick
        track.append(message)
        previous_tick = absolute_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def _whole_note_fraction_to_ticks(value: Fraction, *, ticks_per_beat: int) -> int:
    return round(float(value * _QUARTERS_PER_WHOLE * ticks_per_beat))
