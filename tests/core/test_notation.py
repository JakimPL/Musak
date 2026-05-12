import pytest
from pydantic import ValidationError

from musak.core.notation.chord_serializer import (
    interval_to_score_data,
    inversion_to_score_data,
)
from musak.core.notation.rhythm_serializer import (
    note_to_note_data,
    phrases_to_score_data,
    split_into_measures,
)
from musak.core.notation.schema import NoteData, ScoreData, StaveData, VoiceData
from musak.modules.elements.interval import Interval
from musak.modules.elements.inversion import ChordInversion
from musak.modules.elements.names import midi_to_vexflow_key
from musak.modules.elements.note import Note
from musak.modules.elements.phrase import Phrase
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
            ScoreData(rows=[], tempo=0)

        with pytest.raises(ValidationError):
            ScoreData(rows=[], tempo=-1)

    def test_accepts_none_tempo(self) -> None:
        score = ScoreData(rows=[], tempo=None)
        assert score.tempo is None


class TestIntervalToScoreData:
    def test_chord_mode_produces_whole_note(self) -> None:
        interval = Interval(interval=7, base_note_index=60)
        score = interval_to_score_data(interval, sequential=False, tempo=120)

        assert len(score.rows) == 1
        assert len(score.rows[0]) == 1
        voice = score.rows[0][0].voices[0]
        assert voice.notes[0].duration == "w"
        assert "c/4" in voice.notes[0].keys
        assert "g/4" in voice.notes[0].keys

    def test_sequential_mode_produces_quarter_notes(self) -> None:
        interval = Interval(interval=7, base_note_index=60)
        score = interval_to_score_data(interval, sequential=True, tempo=120)

        voice = score.rows[0][0].voices[0]
        assert len(voice.notes) == 2
        assert voice.notes[0].duration == "q"
        assert voice.notes[1].duration == "q"

    def test_low_notes_uses_bass_clef(self) -> None:
        interval = Interval(interval=4, base_note_index=40)
        score = interval_to_score_data(interval, sequential=False, tempo=120)
        assert score.rows[0][0].clef == "bass"

    def test_treble_notes_uses_treble_clef(self) -> None:
        interval = Interval(interval=4, base_note_index=65)
        score = interval_to_score_data(interval, sequential=False, tempo=120)
        assert score.rows[0][0].clef == "treble"

    def test_wide_span_uses_treble_clef(self) -> None:
        interval = Interval(interval=25, base_note_index=40)
        score = interval_to_score_data(interval, sequential=False, tempo=120)
        assert score.rows[0][0].clef == "treble"

    def test_tempo_is_propagated(self) -> None:
        interval = Interval(interval=5, base_note_index=60)
        score = interval_to_score_data(interval, sequential=False, tempo=80)
        assert score.tempo == 80

    def test_no_time_signature(self) -> None:
        interval = Interval(interval=5, base_note_index=60)
        score = interval_to_score_data(interval, sequential=False, tempo=120)
        assert score.rows[0][0].time_signature is None


class TestInversionToScoreData:
    def test_produces_whole_note_chord(self) -> None:
        inversion = ChordInversion(chord_type="", base_chord=(0, 4, 7), inversion_index=0, base_note_index=60)
        score = inversion_to_score_data(inversion, sequential=False, tempo=120)

        voice = score.rows[0][0].voices[0]
        assert voice.notes[0].duration == "w"
        assert len(voice.notes[0].keys) == 3

    def test_low_chord_uses_bass_clef(self) -> None:
        inversion = ChordInversion(chord_type="m", base_chord=(0, 3, 7), inversion_index=0, base_note_index=40)
        score = inversion_to_score_data(inversion, sequential=False, tempo=120)
        assert score.rows[0][0].clef == "bass"

    def test_no_time_signature(self) -> None:
        inversion = ChordInversion(chord_type="", base_chord=(0, 4, 7), inversion_index=0, base_note_index=60)
        score = inversion_to_score_data(inversion, sequential=False, tempo=120)
        assert score.rows[0][0].time_signature is None


class TestNoteToNoteData:
    def test_quarter_note(self) -> None:
        note = Note(duration=4, pause=False)  # Fraction(1, 4) = quarter
        data = note_to_note_data(note)
        assert data.duration == "q"
        assert data.dots == 0
        assert data.keys == ["b/4"]

    def test_quarter_rest(self) -> None:
        note = Note(duration=-4)
        data = note_to_note_data(note)
        assert data.duration == "qr"
        assert data.dots == 0

    def test_dotted_eighth(self) -> None:
        from fractions import Fraction

        note = Note(duration=Fraction(3, 16))  # dotted eighth
        data = note_to_note_data(note)
        assert data.duration == "8"
        assert data.dots == 1

    def test_whole_note(self) -> None:
        note = Note(duration=1)  # 1/1 = whole
        data = note_to_note_data(note)
        assert data.duration == "w"

    def test_sixteenth_rest(self) -> None:
        note = Note(duration=-16)
        data = note_to_note_data(note)
        assert data.duration == "16r"


class TestSplitIntoMeasures:
    def test_single_measure_4_4(self) -> None:
        phrase = Phrase(notes=[Note(duration=4)] * 4)  # four quarter notes
        measures = split_into_measures(phrase, (4, 4))
        assert len(measures) == 1
        assert len(measures[0].notes) == 4

    def test_two_measures_4_4(self) -> None:
        phrase = Phrase(notes=[Note(duration=4)] * 8)  # eight quarter notes
        measures = split_into_measures(phrase, (4, 4))
        assert len(measures) == 2
        assert all(len(m.notes) == 4 for m in measures)

    def test_three_four_time(self) -> None:
        phrase = Phrase(notes=[Note(duration=4)] * 6)  # six quarter notes
        measures = split_into_measures(phrase, (3, 4))
        assert len(measures) == 2
        assert all(len(m.notes) == 3 for m in measures)


class TestPhrasesToScoreData:
    def test_single_group_single_measure(self) -> None:
        phrase = Phrase(notes=[Note(duration=4)] * 4)
        score = phrases_to_score_data([phrase], time_signature=(4, 4), tempo=120)
        assert len(score.rows) == 1
        assert len(score.rows[0]) == 1
        assert score.rows[0][0].clef == "percussion"
        assert score.rows[0][0].time_signature == (4, 4)
        assert score.tempo == 120

    def test_first_measure_shows_time_signature(self) -> None:
        phrase = Phrase(notes=[Note(duration=4)] * 8)  # two 4/4 measures
        score = phrases_to_score_data([phrase], time_signature=(4, 4), tempo=100)
        assert score.rows[0][0].time_signature == (4, 4)
        assert score.rows[0][1].time_signature is None

    def test_two_groups_two_measures(self) -> None:
        phrase = Phrase(notes=[Note(duration=4)] * 8)
        score = phrases_to_score_data([phrase, phrase], time_signature=(4, 4), tempo=80)
        assert len(score.rows) == 2
        assert len(score.rows[0]) == 2
        assert len(score.rows[1]) == 2
