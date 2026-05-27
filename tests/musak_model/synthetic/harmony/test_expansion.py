from dataclasses import dataclass

import pytest

from musak_model.synthetic.harmony.expansion import (
    UnspellableChordError,
    expand_chord_to_tones,
)
from musak_model.synthetic.harmony.schema import Chord, ChordExtension, ChordQuality
from musak_model.synthetic.harmony.vocabulary import ChordVocabularyConfig
from musak_model.tokens.schema import ScaleType


@pytest.fixture(scope="module")
def vocabulary() -> ChordVocabularyConfig:
    return ChordVocabularyConfig.load()


@dataclass(frozen=True)
class ExpansionCase:
    identifier: str
    chord: Chord
    scale_type: ScaleType
    expected_tones: frozenset[tuple[int, int]]


_EXPANSION_CASES: tuple[ExpansionCase, ...] = (
    ExpansionCase(
        identifier="major_tonic",
        chord=Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
        scale_type=ScaleType.MAJOR,
        expected_tones=frozenset({(1, 0), (3, 0), (5, 0)}),
    ),
    ExpansionCase(
        identifier="borrowed_minor_iv",
        chord=Chord(root_degree=4, root_accidental=0, quality=ChordQuality.MINOR),
        scale_type=ScaleType.MAJOR,
        expected_tones=frozenset({(4, 0), (6, -1), (1, 0)}),
    ),
    ExpansionCase(
        identifier="secondary_dominant_of_dominant",
        chord=Chord(root_degree=2, root_accidental=0, quality=ChordQuality.MAJOR),
        scale_type=ScaleType.MAJOR,
        expected_tones=frozenset({(2, 0), (4, 1), (6, 0)}),
    ),
    ExpansionCase(
        identifier="flat_six_major",
        chord=Chord(root_degree=6, root_accidental=-1, quality=ChordQuality.MAJOR),
        scale_type=ScaleType.MAJOR,
        expected_tones=frozenset({(6, -1), (1, 0), (3, -1)}),
    ),
    ExpansionCase(
        identifier="leading_tone_diminished",
        chord=Chord(root_degree=7, root_accidental=0, quality=ChordQuality.DIMINISHED),
        scale_type=ScaleType.MAJOR,
        expected_tones=frozenset({(7, 0), (2, 0), (4, 0)}),
    ),
    ExpansionCase(
        identifier="harmonic_minor_dominant",
        chord=Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR),
        scale_type=ScaleType.HARMONIC_MINOR,
        expected_tones=frozenset({(5, 0), (7, 0), (2, 0)}),
    ),
    ExpansionCase(
        identifier="dominant_seventh",
        chord=Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR, extension=ChordExtension.SEVENTH),
        scale_type=ScaleType.MAJOR,
        expected_tones=frozenset({(5, 0), (7, 0), (2, 0), (4, 0)}),
    ),
    ExpansionCase(
        identifier="major_seventh",
        chord=Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR, extension=ChordExtension.SEVENTH),
        scale_type=ScaleType.MAJOR,
        expected_tones=frozenset({(1, 0), (3, 0), (5, 0), (7, 0)}),
    ),
)


@pytest.mark.parametrize("case", _EXPANSION_CASES, ids=lambda case: case.identifier)
def test_expand_chord_to_tones(case: ExpansionCase, vocabulary: ChordVocabularyConfig) -> None:
    tones = expand_chord_to_tones(case.chord, scale_type=case.scale_type, vocabulary=vocabulary)

    assert {(tone.degree, tone.accidental) for tone in tones} == case.expected_tones


def test_expand_chord_to_tones_preserves_member_count(vocabulary: ChordVocabularyConfig) -> None:
    chord = Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR, extension=ChordExtension.SEVENTH)

    tones = expand_chord_to_tones(chord, scale_type=ScaleType.MAJOR, vocabulary=vocabulary)

    assert len(tones) == 4


def test_expand_chord_to_tones_raises_on_double_accidental(vocabulary: ChordVocabularyConfig) -> None:
    chord = Chord(root_degree=6, root_accidental=0, quality=ChordQuality.AUGMENTED)

    with pytest.raises(UnspellableChordError):
        expand_chord_to_tones(chord, scale_type=ScaleType.MELODIC_MINOR, vocabulary=vocabulary)


def test_expand_chord_to_tones_raises_when_root_exceeds_scale_size(vocabulary: ChordVocabularyConfig) -> None:
    chord = Chord(root_degree=8, root_accidental=0, quality=ChordQuality.MAJOR)

    with pytest.raises(ValueError, match="exceeds"):
        expand_chord_to_tones(chord, scale_type=ScaleType.MAJOR, vocabulary=vocabulary)
