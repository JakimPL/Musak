from musak.core.notation.chord_serializer import (
    interval_to_score_data,
    inversion_to_score_data,
)
from musak.core.notation.rhythm_serializer import (
    note_to_note_data,
    phrases_to_score_data,
    split_into_measures,
)
from musak.modules.elements.interval import Interval
from musak.modules.elements.inversion import ChordInversion
from musak.modules.elements.note import Note
from musak.modules.elements.phrase import Phrase


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
