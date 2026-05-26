from musak.core.notation.chord_serializer import interval_to_score_data, inversion_to_score_data
from musak.modules.elements.interval import Interval
from musak.modules.elements.inversion import ChordInversion


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
