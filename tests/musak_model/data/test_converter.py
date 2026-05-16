from dataclasses import dataclass

import pytest

from musak_model.data.converter import pitch_to_degree
from musak_model.tokens.schema import Hand, ScaleType


@dataclass(frozen=True)
class DegreeTestCase:
    midi_pitch: int
    key_root: int
    key_fifths: int
    scale_type: ScaleType
    hand: Hand
    expected_degree: int
    expected_accidental: int
    expected_octave_offset: int | None = None


TEST_CASES = [
    DegreeTestCase(60, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 1, 0, -1),  # C major tonic
    DegreeTestCase(61, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 1, 1, -1),  # C# in C major
    DegreeTestCase(61, 3, -3, ScaleType.MAJOR, Hand.RIGHT, 7, -1, -1),  # Db in Eb major
    DegreeTestCase(62, 2, 2, ScaleType.MAJOR, Hand.RIGHT, 1, 0, -1),  # D major tonic
    DegreeTestCase(61, 7, 1, ScaleType.MAJOR, Hand.RIGHT, 4, 1, -1),  # C# in G major
    DegreeTestCase(66, 10, -2, ScaleType.MAJOR, Hand.RIGHT, 6, -1, -1),  # Gb in Bb major
    DegreeTestCase(60, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 1, 0, -1),  # C4 RH
    DegreeTestCase(60, 0, 0, ScaleType.MAJOR, Hand.LEFT, 1, 0, 1),  # C4 LH
]


@pytest.mark.parametrize("case", TEST_CASES)
def test_pitch_to_degree_parametrized(case: DegreeTestCase) -> None:
    result = pitch_to_degree(
        case.midi_pitch,
        key_root=case.key_root,
        key_fifths=case.key_fifths,
        scale_type=case.scale_type,
        hand=case.hand,
    )
    assert result.degree == case.expected_degree, f"degree: {result.degree} != {case.expected_degree}"
    assert (
        result.accidental == case.expected_accidental
    ), f"accidental: {result.accidental} != {case.expected_accidental}"
    if case.expected_octave_offset is not None:
        assert (
            result.octave_offset == case.expected_octave_offset
        ), f"octave_offset: {result.octave_offset} != {case.expected_octave_offset}"
