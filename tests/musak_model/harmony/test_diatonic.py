from dataclasses import dataclass

import pytest

from musak_model.harmony.diatonic import natural_triad
from musak_model.harmony.schema import ChordQuality
from musak_model.tokens.schema import ScaleType


@dataclass(frozen=True)
class _TriadCase:
    scale_type: ScaleType
    degree: int
    expected_quality: ChordQuality


_TRIAD_CASES = (
    _TriadCase(ScaleType.MAJOR, 1, ChordQuality.MAJOR),
    _TriadCase(ScaleType.MAJOR, 2, ChordQuality.MINOR),
    _TriadCase(ScaleType.MAJOR, 5, ChordQuality.MAJOR),
    _TriadCase(ScaleType.MAJOR, 6, ChordQuality.MINOR),
    _TriadCase(ScaleType.MAJOR, 7, ChordQuality.DIMINISHED),
    _TriadCase(ScaleType.HARMONIC_MINOR, 1, ChordQuality.MINOR),
    _TriadCase(ScaleType.HARMONIC_MINOR, 5, ChordQuality.MAJOR),
    _TriadCase(ScaleType.HARMONIC_MINOR, 7, ChordQuality.DIMINISHED),
)


@pytest.mark.parametrize("case", _TRIAD_CASES, ids=lambda case: f"{case.scale_type.value}-{case.degree}")
def test_natural_triad_quality_follows_the_scale(case: _TriadCase) -> None:
    triad = natural_triad(case.scale_type, case.degree)

    assert triad.root_degree == case.degree
    assert triad.root_accidental == 0
    assert triad.quality is case.expected_quality
