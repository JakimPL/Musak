from fractions import Fraction

from musak_model.decoder import PianoRollEvent
from musak_model.tokens.schema import Hand
from notebooks.utils.audio import midi_file_bytes, piano_roll_events_to_audio_data, piano_roll_events_to_midi_file


class FakeExporter:
    def __init__(self) -> None:
        self.midi_data = b""

    def export_audio(self, midi_data: bytes) -> str:
        self.midi_data = midi_data
        return "data:audio/mpeg;base64,abc"


def test_piano_roll_events_to_midi_file_writes_tempo_and_hand_tracks() -> None:
    midi_file = piano_roll_events_to_midi_file(
        [
            PianoRollEvent(hand=Hand.RIGHT, midi_pitch=60, start=Fraction(0), duration=Fraction(1, 4)),
            PianoRollEvent(hand=Hand.LEFT, midi_pitch=48, start=Fraction(1, 4), duration=Fraction(1, 4)),
        ],
        bpm=120,
    )

    assert midi_file.ticks_per_beat == 480
    assert midi_file.tracks[0][0].type == "set_tempo"
    assert midi_file.tracks[0][0].tempo == 500000
    assert [track[0].type for track in midi_file.tracks[1:]] == ["program_change", "program_change"]
    assert _note_messages(midi_file.tracks[1]) == [
        ("note_on", 60, 0),
        ("note_off", 60, 480),
    ]
    assert _note_messages(midi_file.tracks[2]) == [
        ("note_on", 48, 480),
        ("note_off", 48, 480),
    ]


def test_piano_roll_events_to_midi_file_filters_hands() -> None:
    midi_file = piano_roll_events_to_midi_file(
        [
            PianoRollEvent(hand=Hand.RIGHT, midi_pitch=60, start=Fraction(0), duration=Fraction(1, 4)),
            PianoRollEvent(hand=Hand.LEFT, midi_pitch=48, start=Fraction(0), duration=Fraction(1, 4)),
        ],
        bpm=60,
        hands=frozenset({Hand.LEFT}),
    )

    assert len(midi_file.tracks) == 2
    assert [message.note for message in midi_file.tracks[1] if message.type == "note_on"] == [48]


def test_piano_roll_events_to_midi_file_keeps_chord_onsets_simultaneous() -> None:
    midi_file = piano_roll_events_to_midi_file(
        [
            PianoRollEvent(hand=Hand.RIGHT, midi_pitch=60, start=Fraction(0), duration=Fraction(1, 4)),
            PianoRollEvent(hand=Hand.RIGHT, midi_pitch=64, start=Fraction(0), duration=Fraction(1, 4)),
        ],
        bpm=60,
    )

    note_on_times = [message.time for message in midi_file.tracks[1] if message.type == "note_on"]

    assert note_on_times == [0, 0]


def test_piano_roll_events_to_audio_data_delegates_rendering_to_exporter() -> None:
    exporter = FakeExporter()

    audio_data = piano_roll_events_to_audio_data(
        [PianoRollEvent(hand=Hand.RIGHT, midi_pitch=60, start=Fraction(0), duration=Fraction(1, 4))],
        bpm=60,
        exporter=exporter,
    )

    assert audio_data == "data:audio/mpeg;base64,abc"
    assert midi_file_bytes(piano_roll_events_to_midi_file([], bpm=60)).startswith(b"MThd")
    assert exporter.midi_data.startswith(b"MThd")


def _note_messages(track) -> list[tuple[str, int, int]]:
    return [(message.type, message.note, message.time) for message in track if message.type in {"note_on", "note_off"}]
