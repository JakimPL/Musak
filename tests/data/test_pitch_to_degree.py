from dataclasses import dataclass

import pytest

from musak_model.data.converter import pitch_to_degree
from musak_model.tokens.schema import Hand, ScaleType


@dataclass(frozen=True)
class PitchToDegreeCase:
    midi_pitch: int
    key_root: int
    key_fifths: int
    scale_type: ScaleType
    hand: Hand
    expected_degree: int
    expected_accidental: int
    expected_octave_offset: int | None = None


class TestPitchToDegreeMAJOR:
    MAJOR_CASES = [
        PitchToDegreeCase(60, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 1, 0, -1),  # C
        PitchToDegreeCase(62, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 2, 0, -1),  # D
        PitchToDegreeCase(64, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 3, 0, -1),  # E
        PitchToDegreeCase(65, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 4, 0, -1),  # F
        PitchToDegreeCase(67, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 5, 0, -1),  # G
        PitchToDegreeCase(69, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 6, 0, -1),  # A
        PitchToDegreeCase(71, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 7, 0, -1),  # B
        PitchToDegreeCase(61, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 1, 1, -1),  # C#
        PitchToDegreeCase(63, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 2, 1, -1),  # D#
        PitchToDegreeCase(61, 7, 1, ScaleType.MAJOR, Hand.RIGHT, 4, 1, -1),  # C# in G major (F#=degree 4)
        PitchToDegreeCase(66, 10, -2, ScaleType.MAJOR, Hand.RIGHT, 6, -1, -1),  # Gb in Bb major (degree 6 = A -> Ab)
        PitchToDegreeCase(60, 0, 0, ScaleType.MAJOR, Hand.RIGHT, 1, 0, -1),  # C4 RH (octave 4 - 5 = -1)
        PitchToDegreeCase(60, 0, 0, ScaleType.MAJOR, Hand.LEFT, 1, 0, 1),  # C4 LH (octave 4 - 3 = 1)
        PitchToDegreeCase(48, 0, 0, ScaleType.MAJOR, Hand.LEFT, 1, 0, 0),  # C3 LH (octave 3 - 3 = 0)
    ]

    @pytest.mark.parametrize("case", MAJOR_CASES)
    def test_major_scale_mapping(self, case: PitchToDegreeCase) -> None:
        result = pitch_to_degree(
            case.midi_pitch,
            key_root=case.key_root,
            key_fifths=case.key_fifths,
            scale_type=case.scale_type,
            hand=case.hand,
        )

        assert result.degree == case.expected_degree, (
            f"degree for pitch {case.midi_pitch} in {case.scale_type.value}: "
            f"{result.degree} != {case.expected_degree}"
        )
        assert result.accidental == case.expected_accidental, (
            f"accidental for pitch {case.midi_pitch} in {case.scale_type.value}: "
            f"{result.accidental} != {case.expected_accidental}"
        )
        if case.expected_octave_offset is not None:
            assert result.octave_offset == case.expected_octave_offset, (
                f"octave_offset for pitch {case.midi_pitch} hand {case.hand.value}: "
                f"{result.octave_offset} != {case.expected_octave_offset}"
            )


class TestPitchToDegreeHARMONIC_MINOR:
    HARMONIC_MINOR_CASES = [
        PitchToDegreeCase(69, 9, 0, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 1, 0, 0),  # A
        PitchToDegreeCase(71, 9, 0, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 2, 0, 0),  # B
        PitchToDegreeCase(60, 9, 0, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 3, 0, -1),  # C (octave down)
        PitchToDegreeCase(62, 9, 0, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 4, 0, 0),  # D
        PitchToDegreeCase(64, 9, 0, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 5, 0, 0),  # E
        PitchToDegreeCase(65, 9, 0, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 6, 0, 0),  # F
        PitchToDegreeCase(68, 9, 0, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 7, 0, 0),  # G# (raised 7th)
        PitchToDegreeCase(64, 4, 1, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 1, 0, 0),  # E
        PitchToDegreeCase(66, 4, 1, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 2, 0, 0),  # F#
        PitchToDegreeCase(67, 4, 1, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 3, 0, 0),  # G
        PitchToDegreeCase(69, 4, 1, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 4, 0, 0),  # A
        PitchToDegreeCase(71, 4, 1, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 5, 0, 0),  # B
        PitchToDegreeCase(60, 4, 1, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 6, 0, -1),  # C
        PitchToDegreeCase(63, 4, 1, ScaleType.HARMONIC_MINOR, Hand.RIGHT, 7, 0, 0),  # D#
    ]

    @pytest.mark.parametrize("case", HARMONIC_MINOR_CASES)
    def test_harmonic_minor_scale_mapping(self, case: PitchToDegreeCase) -> None:
        result = pitch_to_degree(
            case.midi_pitch,
            key_root=case.key_root,
            key_fifths=case.key_fifths,
            scale_type=case.scale_type,
            hand=case.hand,
        )

        assert result.degree == case.expected_degree, (
            f"degree for pitch {case.midi_pitch} in harmonic minor (root={case.key_root}): "
            f"{result.degree} != {case.expected_degree}"
        )
        assert result.accidental == case.expected_accidental, (
            f"accidental for pitch {case.midi_pitch} in harmonic minor (root={case.key_root}): "
            f"{result.accidental} != {case.expected_accidental}"
        )


class TestPitchToDegreeEnharmonicPolicy:
    def test_flat_preference_negative_key_fifths(self) -> None:
        result = pitch_to_degree(
            61,  # C#/Db
            key_root=0,
            key_fifths=-2,
            scale_type=ScaleType.MAJOR,
            hand=Hand.RIGHT,
        )
        assert result.degree == 2
        assert result.accidental == -1

    def test_sharp_preference_positive_key_fifths(self) -> None:
        result = pitch_to_degree(
            61,
            key_root=7,
            key_fifths=1,
            scale_type=ScaleType.MAJOR,
            hand=Hand.RIGHT,
        )

        assert result.degree == 4
        assert result.accidental == 1


class TestPitchToDegreeEdgeCases:
    def test_chromatic_pitch_in_scale_maps_to_accidental(self) -> None:
        result = pitch_to_degree(
            61,  # C#/Db
            key_root=0,
            key_fifths=0,
            scale_type=ScaleType.MAJOR,
            hand=Hand.RIGHT,
        )
        assert 1 <= result.degree <= 7
        assert result.accidental != 0

    def test_low_octave_negative_offset(self) -> None:
        result = pitch_to_degree(
            36,  # C2
            key_root=0,
            key_fifths=0,
            scale_type=ScaleType.MAJOR,
            hand=Hand.LEFT,
        )
        assert result.octave_offset < 0

    def test_high_octave_positive_offset(self) -> None:
        result = pitch_to_degree(
            84,  # C6
            key_root=0,
            key_fifths=0,
            scale_type=ScaleType.MAJOR,
            hand=Hand.RIGHT,
        )
        assert result.octave_offset > 0
