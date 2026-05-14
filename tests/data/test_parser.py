from music21 import key as m21_key
from music21.meter.base import TimeSignature
from music21.note import Note
from music21.stream.base import Measure

from musak_model.data.parser import _parse_measure


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
