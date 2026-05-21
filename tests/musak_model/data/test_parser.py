from fractions import Fraction
from pathlib import Path

import pytest
from music21 import chord, instrument
from music21 import key as m21_key
from music21 import tie
from music21.meter.base import TimeSignature
from music21.note import Note, Rest
from music21.stream.base import Measure, Part, Score

from musak_model.data.parser import _detect_key, _parse_measure, _tie_type_from_text, parse_score
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
        assert parsed.declared_key_fifths == 2
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

    def test_captures_actual_short_measure_duration_without_shifting_notes(self) -> None:
        measure = Measure(number=0)
        measure.insert(0, TimeSignature("4/4"))
        measure.insert(0, Note("C4", quarterLength=1))

        parsed = _parse_measure(measure, default_time_signature=(4, 4), default_key_fifths=0)

        assert parsed.time_numerator == 4
        assert parsed.time_denominator == 4
        assert parsed.measure_duration == Fraction(1, 4)
        parsed_note = parsed.events[0]
        assert isinstance(parsed_note, ParsedNote)
        assert parsed_note.beat_offset == 0


class TestTieParsing:
    @pytest.mark.parametrize(
        ("music21_tie_type", "expected"),
        [
            ("start", TieType.START),
            ("continue", TieType.CONTINUE),
            ("stop", TieType.STOP),
        ],
    )
    def test_preserves_note_tie_type(self, music21_tie_type: str, expected: TieType) -> None:
        tied_note = Note("C4", quarterLength=1)
        tied_note.tie = tie.Tie(music21_tie_type)
        measure = Measure()
        measure.insert(0, TimeSignature("4/4"))
        measure.insert(0, tied_note)

        parsed = _parse_measure(measure, default_time_signature=(4, 4), default_key_fifths=0)

        parsed_note = parsed.events[0]
        assert isinstance(parsed_note, ParsedNote)
        assert parsed_note.tie_type == expected

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

    def test_marks_mixed_chord_tie_types_as_partial(self) -> None:
        tied_chord = chord.Chord(["C4", "E4"], quarterLength=1)
        tied_chord.notes[0].tie = tie.Tie("start")
        tied_chord.notes[1].tie = tie.Tie("stop")
        measure = Measure()
        measure.insert(0, TimeSignature("4/4"))
        measure.insert(0, tied_chord)

        parsed = _parse_measure(measure, default_time_signature=(4, 4), default_key_fifths=0)

        parsed_chord = parsed.events[0]
        assert isinstance(parsed_chord, ParsedChord)
        assert parsed_chord.tie_type == TieType.PARTIAL

    @pytest.mark.parametrize(
        ("music21_tie_type", "expected"),
        [
            ("start", TieType.START),
            ("continue", TieType.CONTINUE),
            ("stop", TieType.STOP),
        ],
    )
    def test_preserves_full_chord_tie_type(self, music21_tie_type: str, expected: TieType) -> None:
        tied_chord = chord.Chord(["C4", "E4"], quarterLength=1)
        for chord_note in tied_chord.notes:
            chord_note.tie = tie.Tie(music21_tie_type)
        measure = Measure()
        measure.insert(0, TimeSignature("4/4"))
        measure.insert(0, tied_chord)

        parsed = _parse_measure(measure, default_time_signature=(4, 4), default_key_fifths=0)

        parsed_chord = parsed.events[0]
        assert isinstance(parsed_chord, ParsedChord)
        assert parsed_chord.tie_type == expected

    def test_rejects_unknown_tie_type(self) -> None:
        with pytest.raises(ValueError, match="unsupported tie type"):
            _tie_type_from_text("let-ring")


class TestScoreParsing:
    def test_parse_score_preserves_cross_measure_ties(self, tmp_path: Path) -> None:
        score = Score()
        right = Part()
        right.insert(0, instrument.Piano())
        left = Part()
        left.insert(0, instrument.Piano())

        right_measure_1 = Measure(number=1)
        right_measure_1.insert(0, TimeSignature("4/4"))
        right_note_start = Note("C5", quarterLength=4)
        right_note_start.tie = tie.Tie("start")
        right_measure_1.insert(0, right_note_start)
        right.append(right_measure_1)

        right_measure_2 = Measure(number=2)
        right_note_stop = Note("C5", quarterLength=2)
        right_note_stop.tie = tie.Tie("stop")
        right_measure_2.insert(0, right_note_stop)
        right.append(right_measure_2)

        left_measure_1 = Measure(number=1)
        left_measure_1.insert(0, TimeSignature("4/4"))
        left_measure_1.insert(0, Note("C3", quarterLength=1))
        left.append(left_measure_1)

        left_measure_2 = Measure(number=2)
        left_measure_2.insert(0, Note("C3", quarterLength=1))
        left.append(left_measure_2)

        score.insert(0, left)
        score.insert(0, right)
        path = tmp_path / "tied.musicxml"
        score.write("musicxml", fp=path)

        parsed = parse_score(path)

        assert isinstance(parsed.right_hand_bars[0].events[0], ParsedNote)
        assert isinstance(parsed.right_hand_bars[1].events[0], ParsedNote)
        assert parsed.right_hand_bars[0].events[0].tie_type == TieType.START
        assert parsed.right_hand_bars[1].events[0].tie_type == TieType.STOP


class TestKeyDetection:
    def test_uses_explicit_key_signature_as_declared_pitch_set(self) -> None:
        score = Score()
        part = Part()
        measure = Measure(number=1)
        measure.insert(0, m21_key.KeySignature(0))
        measure.insert(0, Note("D4", quarterLength=1))
        measure.insert(1, Note("F4", quarterLength=1))
        part.append(measure)
        score.insert(0, part)

        assert _detect_key(score) == 0

    def test_missing_key_signature_has_no_declared_pitch_set(self) -> None:
        score = Score()
        part = Part()
        measure = Measure(number=1)
        measure.insert(0, Note("D4", quarterLength=1))
        part.append(measure)
        score.insert(0, part)

        assert _detect_key(score) is None

    def test_minor_key_signature_keeps_major_parent_pitch_set(self) -> None:
        score = Score()
        part = Part()
        measure = Measure(number=1)
        key_signature = m21_key.KeySignature(2)
        key_signature.mode = "minor"
        measure.insert(0, key_signature)
        part.append(measure)
        score.insert(0, part)

        assert _detect_key(score) == 2
