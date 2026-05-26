from fractions import Fraction

from musak_model.data.schema import ParsedNote


def test_parsed_events_accept_zero_beat_offset() -> None:
    assert ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0)).beat_offset == 0
