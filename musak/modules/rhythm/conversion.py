import pathlib
from fractions import Fraction

import mido

from musak.config.defaults import MELODIC, TEMPO, TIME_SIGNATURE
from musak.modules.elements.constants import (
    MIDI_MELODIC_CHANNEL,
    MIDI_MELODIC_NOTE,
    MIDI_PERCUSSION_CHANNEL,
    MIDI_PERCUSSION_NOTE,
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


def phrases_to_midi(
    phrases: list[Phrase],
    *,
    time_signature: TimeSignatureType = TIME_SIGNATURE,
    tempo: int = TEMPO,
    melodic: bool = MELODIC,
) -> mido.MidiFile:
    if not phrases:
        raise EmptyScoreException("an empty score")

    for phrase in phrases:
        invalid_beat = phrase.find_invalid_beat(time_signature=time_signature)
        if invalid_beat:
            raise InvalidBeatException(f"invalid beat no. {invalid_beat}")

    mid = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    num, den = time_signature
    track.append(mido.MetaMessage("set_tempo", tempo=_tempo_to_us(tempo), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0))

    channel = MIDI_MELODIC_CHANNEL if melodic else MIDI_PERCUSSION_CHANNEL
    note = MIDI_MELODIC_NOTE if melodic else MIDI_PERCUSSION_NOTE

    for phrase in phrases:
        for n in phrase.notes:
            ticks = _note_ticks(n.duration)
            if n.pause:
                track.append(mido.Message("note_on", channel=channel, note=note, velocity=0, time=0))
                track.append(mido.Message("note_off", channel=channel, note=note, velocity=0, time=ticks))
            else:
                track.append(mido.Message("note_on", channel=channel, note=note, velocity=MIDI_VELOCITY, time=0))
                track.append(mido.Message("note_off", channel=channel, note=note, velocity=0, time=ticks))

    return mid


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
