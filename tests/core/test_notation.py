import pytest
from pydantic import ValidationError

from musak.core.notation.chord_serializer import (
    interval_to_score_data,
    inversion_to_score_data,
)
from musak.core.notation.schema import NoteData, ScoreData, StaveData, VoiceData
from musak.modules.elements.interval import Interval
from musak.modules.elements.inversion import ChordInversion
from musak.modules.elements.names import midi_to_vexflow_key
from musak.modules.elements.time_signature import validate_time_signature


class TestMidiToVexflowKey:
    def test_middle_c(self) -> None:
        assert midi_to_vexflow_key(60) == "c/4"

    def test_sharp(self) -> None:
        assert midi_to_vexflow_key(61) == "c#/4"

    def test_below_middle_c(self) -> None:
        assert midi_to_vexflow_key(59) == "b/3"

    def test_octave_boundary(self) -> None:
        assert midi_to_vexflow_key(48) == "c/3"
        assert midi_to_vexflow_key(72) == "c/5"


class TestValidateTimeSignature:
    def test_accepts_valid(self) -> None:
        assert validate_time_signature((4, 4)) == (4, 4)
        assert validate_time_signature((3, 8)) == (3, 8)
        assert validate_time_signature((7, 16)) == (7, 16)

    def test_rejects_zero_numerator(self) -> None:
        with pytest.raises(ValueError):
            validate_time_signature((0, 4))

    def test_rejects_negative_numerator(self) -> None:
        with pytest.raises(ValueError):
            validate_time_signature((-1, 4))

    def test_rejects_non_power_of_two_denominator(self) -> None:
        with pytest.raises(ValueError):
            validate_time_signature((4, 3))

    def test_rejects_zero_denominator(self) -> None:
        with pytest.raises(ValueError):
            validate_time_signature((4, 0))


class TestNoteData:
    def test_rejects_invalid_duration(self) -> None:
        with pytest.raises(ValidationError):
            NoteData(keys=["c/4"], duration="xyz")

    def test_rejects_negative_dots(self) -> None:
        with pytest.raises(ValidationError):
            NoteData(keys=["c/4"], duration="w", dots=-1)

    def test_rejects_too_many_dots(self) -> None:
        with pytest.raises(ValidationError):
            NoteData(keys=["c/4"], duration="w", dots=3)

    def test_accepts_rest_duration(self) -> None:
        note = NoteData(keys=[], duration="wr")
        assert note.duration == "wr"
        assert note.keys == []


class TestStaveData:
    def test_rejects_invalid_clef(self) -> None:
        with pytest.raises(ValidationError):
            StaveData(clef="unknown", voices=[])

    def test_rejects_invalid_time_signature(self) -> None:
        with pytest.raises(ValidationError):
            StaveData(clef="treble", time_signature=(4, 3), voices=[])

    def test_accepts_none_time_signature(self) -> None:
        stave = StaveData(clef="treble", time_signature=None, voices=[])
        assert stave.time_signature is None


class TestScoreData:
    def test_rejects_non_positive_tempo(self) -> None:
        with pytest.raises(ValidationError):
            ScoreData(staves=[], tempo=0)

        with pytest.raises(ValidationError):
            ScoreData(staves=[], tempo=-1)

    def test_accepts_none_tempo(self) -> None:
        score = ScoreData(staves=[], tempo=None)
        assert score.tempo is None


class TestIntervalToScoreData:
    def test_chord_mode_produces_whole_note(self) -> None:
        interval = Interval(interval=7, base_note_index=60)
        score = interval_to_score_data(interval, sequential=False, tempo=120)

        assert len(score.staves) == 1
        voice = score.staves[0].voices[0]
        assert voice.notes[0].duration == "w"
        assert "c/4" in voice.notes[0].keys
        assert "g/4" in voice.notes[0].keys

    def test_sequential_mode_produces_quarter_notes(self) -> None:
        interval = Interval(interval=7, base_note_index=60)
        score = interval_to_score_data(interval, sequential=True, tempo=120)

        voice = score.staves[0].voices[0]
        assert len(voice.notes) == 2
        assert voice.notes[0].duration == "q"
        assert voice.notes[1].duration == "q"

    def test_low_notes_uses_bass_clef(self) -> None:
        interval = Interval(interval=4, base_note_index=40)
        score = interval_to_score_data(interval, sequential=False, tempo=120)
        assert score.staves[0].clef == "bass"

    def test_treble_notes_uses_treble_clef(self) -> None:
        interval = Interval(interval=4, base_note_index=65)
        score = interval_to_score_data(interval, sequential=False, tempo=120)
        assert score.staves[0].clef == "treble"

    def test_wide_span_uses_treble_clef(self) -> None:
        interval = Interval(interval=25, base_note_index=40)
        score = interval_to_score_data(interval, sequential=False, tempo=120)
        assert score.staves[0].clef == "treble"

    def test_tempo_is_propagated(self) -> None:
        interval = Interval(interval=5, base_note_index=60)
        score = interval_to_score_data(interval, sequential=False, tempo=80)
        assert score.tempo == 80

    def test_no_time_signature(self) -> None:
        interval = Interval(interval=5, base_note_index=60)
        score = interval_to_score_data(interval, sequential=False, tempo=120)
        assert score.staves[0].time_signature is None


class TestInversionToScoreData:
    def test_produces_whole_note_chord(self) -> None:
        inversion = ChordInversion(chord_type="", base_chord=(0, 4, 7), inversion_index=0, base_note_index=60)
        score = inversion_to_score_data(inversion, tempo=120)

        voice = score.staves[0].voices[0]
        assert voice.notes[0].duration == "w"
        assert len(voice.notes[0].keys) == 3

    def test_low_chord_uses_bass_clef(self) -> None:
        inversion = ChordInversion(chord_type="m", base_chord=(0, 3, 7), inversion_index=0, base_note_index=40)
        score = inversion_to_score_data(inversion, tempo=120)
        assert score.staves[0].clef == "bass"

    def test_no_time_signature(self) -> None:
        inversion = ChordInversion(chord_type="", base_chord=(0, 4, 7), inversion_index=0, base_note_index=60)
        score = inversion_to_score_data(inversion, tempo=120)
        assert score.staves[0].time_signature is None
