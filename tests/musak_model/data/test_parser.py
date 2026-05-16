from music21 import chord
from music21 import key as m21_key
from music21 import tie
from music21.meter.base import TimeSignature
from music21.note import Note, Rest
from music21.stream.base import Measure, Part, Score

from musak_model.data.parser import _detect_key, _parse_measure
from musak_model.data.schema import ParsedChord, ParsedNote, TieType
from musak_model.tokens.schema import ScaleType


class TestMeasureParsing:
    def test_captures_effective_time_and_key_signature(self) -> None:
        measure = Measure()
        measure.insert(0, TimeSignature("3/4"))
        measure.insert(0, m21_key.KeySignature(2))
        measure.insert(0, Note("C4", quarterLength=1))

        parsed = _parse_measure(measure, default_time_signature=(4, 4), default_key_fifths=0)

        assert parsed.time_numerator == 3
        assert parsed.time_denominator == 4
        assert parsed.key_fifths == 2
        assert len(parsed.events) == 1

    def test_skips_non_positive_duration_events(self) -> None:
        measure = Measure()
        zero_note = Note("C4")
        zero_note.duration.quarterLength = 0
        zero_chord = chord.Chord(["E4", "G4"])
        zero_chord.duration.quarterLength = 0
        negative_rest = Rest()
        negative_rest.duration.quarterLength = -1
        measure.insert(0, TimeSignature("4/4"))
        measure.insert(0, zero_note)
        measure.insert(0, zero_chord)
        measure.insert(0, negative_rest)
        measure.insert(0, Note("D4", quarterLength=1))

        parsed = _parse_measure(measure, default_time_signature=(4, 4), default_key_fifths=0)

        assert len(parsed.events) == 1


class TestTieParsing:
    def test_preserves_note_tie_type(self) -> None:
        tied_note = Note("C4", quarterLength=1)
        tied_note.tie = tie.Tie("start")
        measure = Measure()
        measure.insert(0, TimeSignature("4/4"))
        measure.insert(0, tied_note)

        parsed = _parse_measure(measure, default_time_signature=(4, 4), default_key_fifths=0)

        parsed_note = parsed.events[0]
        assert isinstance(parsed_note, ParsedNote)
        assert parsed_note.tie_type == TieType.START

    def test_marks_partial_chord_tie(self) -> None:
        tied_chord = chord.Chord(["C4", "E4"], quarterLength=1)
        tied_chord.notes[0].tie = tie.Tie("start")
        measure = Measure()
        measure.insert(0, TimeSignature("4/4"))
        measure.insert(0, tied_chord)

        parsed = _parse_measure(measure, default_time_signature=(4, 4), default_key_fifths=0)

        parsed_chord = parsed.events[0]
        assert isinstance(parsed_chord, ParsedChord)
        assert parsed_chord.tie_type == TieType.PARTIAL

    def test_preserves_full_chord_tie_type(self) -> None:
        tied_chord = chord.Chord(["C4", "E4"], quarterLength=1)
        for chord_note in tied_chord.notes:
            chord_note.tie = tie.Tie("stop")
        measure = Measure()
        measure.insert(0, TimeSignature("4/4"))
        measure.insert(0, tied_chord)

        parsed = _parse_measure(measure, default_time_signature=(4, 4), default_key_fifths=0)

        parsed_chord = parsed.events[0]
        assert isinstance(parsed_chord, ParsedChord)
        assert parsed_chord.tie_type == TieType.STOP


class TestKeyDetection:
    def test_uses_explicit_key_signature_without_statistical_analysis(self) -> None:
        score = Score()
        part = Part()
        measure = Measure(number=1)
        measure.insert(0, m21_key.KeySignature(0))
        measure.insert(0, Note("D4", quarterLength=1))
        measure.insert(1, Note("F4", quarterLength=1))
        part.append(measure)
        score.insert(0, part)

        assert _detect_key(score) == (0, 0, ScaleType.MAJOR)

    def test_defaults_to_c_major_when_score_has_no_key_signature(self) -> None:
        score = Score()
        part = Part()
        measure = Measure(number=1)
        measure.insert(0, Note("D4", quarterLength=1))
        part.append(measure)
        score.insert(0, part)

        assert _detect_key(score) == (0, 0, ScaleType.MAJOR)

    def test_derives_minor_tonic_from_key_signature_fifths(self) -> None:
        score = Score()
        part = Part()
        measure = Measure(number=1)
        key_signature = m21_key.KeySignature(2)
        key_signature.mode = "minor"
        measure.insert(0, key_signature)
        part.append(measure)
        score.insert(0, part)

        assert _detect_key(score) == (11, 2, ScaleType.HARMONIC_MINOR)
