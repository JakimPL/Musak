import pytest

from musak_model.harmony.diatonic import diatonic_triad, diatonic_triads
from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.tokens.schema import ScaleType


def test_diatonic_triads_return_natural_major_scale_qualities() -> None:
    triads = diatonic_triads(ScaleType.MAJOR)

    assert triads == (
        Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
        Chord(root_degree=2, root_accidental=0, quality=ChordQuality.MINOR),
        Chord(root_degree=3, root_accidental=0, quality=ChordQuality.MINOR),
        Chord(root_degree=4, root_accidental=0, quality=ChordQuality.MAJOR),
        Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR),
        Chord(root_degree=6, root_accidental=0, quality=ChordQuality.MINOR),
        Chord(root_degree=7, root_accidental=0, quality=ChordQuality.DIMINISHED),
    )


def test_diatonic_triad_rejects_degree_outside_scale() -> None:
    with pytest.raises(ValueError, match="root_degree"):
        diatonic_triad(scale_type=ScaleType.MAJOR, root_degree=8)
