import pathlib

import mido

from musak.config.defaults import SEQUENTIAL, TEMPO
from musak.modules.elements.constants import MIDI_TICKS_PER_BEAT, MIDI_VELOCITY


def _tempo_to_us(tempo: int) -> int:
    return 60_000_000 // tempo


def _duration_ticks(beats: float) -> int:
    return round(beats * MIDI_TICKS_PER_BEAT)


def to_midi(
    midi_notes: list[int],
    *,
    tempo: int = TEMPO,
    sequential: bool = SEQUENTIAL,
) -> mido.MidiFile:
    mid = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=_tempo_to_us(tempo), time=0))

    if sequential:
        note_ticks = _duration_ticks(1)
        rest_ticks = _duration_ticks(2)
        for pitch in midi_notes:
            track.append(mido.Message("note_on", note=pitch, velocity=MIDI_VELOCITY, time=0))
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=note_ticks))
        track.append(mido.Message("note_on", note=0, velocity=0, time=0))
        track.append(mido.Message("note_off", note=0, velocity=0, time=rest_ticks))
    else:
        note_ticks = _duration_ticks(4)
        rest_ticks = _duration_ticks(4)
        for pitch in midi_notes:
            track.append(mido.Message("note_on", note=pitch, velocity=MIDI_VELOCITY, time=0))
        for i, pitch in enumerate(midi_notes):
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=note_ticks if i == 0 else 0))
        track.append(mido.Message("note_on", note=0, velocity=0, time=0))
        track.append(mido.Message("note_off", note=0, velocity=0, time=rest_ticks))

    return mid


def save_midi(
    midi_notes: list[int],
    path: pathlib.Path,
    *,
    tempo: int = TEMPO,
    sequential: bool = SEQUENTIAL,
) -> pathlib.Path:
    mid = to_midi(midi_notes, tempo=tempo, sequential=sequential)
    mid.save(str(path))
    return path
