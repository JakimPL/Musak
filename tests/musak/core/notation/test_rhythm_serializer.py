from fractions import Fraction

from musak.core.notation.rhythm_serializer import note_to_note_data, phrases_to_score_data, split_into_measures
from musak.modules.elements.note import Note
from musak.modules.elements.phrase import Phrase


class TestNoteToNoteData:
    def test_quarter_note(self) -> None:
        note = Note(duration=4, pause=False)
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
        note = Note(duration=Fraction(3, 16))
        data = note_to_note_data(note)
        assert data.duration == "8"
        assert data.dots == 1

    def test_whole_note(self) -> None:
        note = Note(duration=1)
        data = note_to_note_data(note)
        assert data.duration == "w"

    def test_sixteenth_rest(self) -> None:
        note = Note(duration=-16)
        data = note_to_note_data(note)
        assert data.duration == "16r"


class TestSplitIntoMeasures:
    def test_single_measure_4_4(self) -> None:
        phrase = Phrase(notes=[Note(duration=4)] * 4)
        measures = split_into_measures(phrase, (4, 4))
        assert len(measures) == 1
        assert len(measures[0].notes) == 4

    def test_two_measures_4_4(self) -> None:
        phrase = Phrase(notes=[Note(duration=4)] * 8)
        measures = split_into_measures(phrase, (4, 4))
        assert len(measures) == 2
        assert all(len(measure.notes) == 4 for measure in measures)

    def test_three_four_time(self) -> None:
        phrase = Phrase(notes=[Note(duration=4)] * 6)
        measures = split_into_measures(phrase, (3, 4))
        assert len(measures) == 2
        assert all(len(measure.notes) == 3 for measure in measures)


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
        phrase = Phrase(notes=[Note(duration=4)] * 8)
        score = phrases_to_score_data([phrase], time_signature=(4, 4), tempo=100)
        assert score.rows[0][0].time_signature == (4, 4)
        assert score.rows[0][1].time_signature is None

    def test_two_groups_two_measures(self) -> None:
        phrase = Phrase(notes=[Note(duration=4)] * 8)
        score = phrases_to_score_data([phrase, phrase], time_signature=(4, 4), tempo=80)
        assert len(score.rows) == 2
        assert len(score.rows[0]) == 2
        assert len(score.rows[1]) == 2
