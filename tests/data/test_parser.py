from music21 import key as m21_key
from music21.meter.base import TimeSignature
from music21.note import Note
from music21.stream.base import Measure, Part, Score

from musak_model.data.parser import _detect_key, _parse_measure
from musak_model.tokens.schema import ScaleType


def test_parse_measure_captures_effective_time_and_key_signature() -> None:
    measure = Measure()
    measure.insert(0, TimeSignature("3/4"))
    measure.insert(0, m21_key.KeySignature(2))
    measure.insert(0, Note("C4", quarterLength=1))

    parsed = _parse_measure(measure, default_time_signature=(4, 4), default_key_fifths=0)

    assert parsed.time_numerator == 3
    assert parsed.time_denominator == 4
    assert parsed.key_fifths == 2
    assert len(parsed.events) == 1


def test_detect_key_uses_explicit_key_signature_without_statistical_analysis() -> None:
    score = Score()
    part = Part()
    measure = Measure(number=1)
    measure.insert(0, m21_key.KeySignature(0))
    measure.insert(0, Note("D4", quarterLength=1))
    measure.insert(1, Note("F4", quarterLength=1))
    part.append(measure)
    score.insert(0, part)

    assert _detect_key(score) == (0, 0, ScaleType.MAJOR)


def test_detect_key_defaults_to_c_major_when_score_has_no_key_signature() -> None:
    score = Score()
    part = Part()
    measure = Measure(number=1)
    measure.insert(0, Note("D4", quarterLength=1))
    part.append(measure)
    score.insert(0, part)

    assert _detect_key(score) == (0, 0, ScaleType.MAJOR)


def test_detect_key_derives_minor_tonic_from_key_signature_fifths() -> None:
    score = Score()
    part = Part()
    measure = Measure(number=1)
    key_signature = m21_key.KeySignature(2)
    key_signature.mode = "minor"
    measure.insert(0, key_signature)
    part.append(measure)
    score.insert(0, part)

    assert _detect_key(score) == (11, 2, ScaleType.HARMONIC_MINOR)
