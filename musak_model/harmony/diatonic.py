from typing import Final

from musak_model.harmony.schema import TRIAD_QUALITY_BY_INTERVALS, Chord
from musak_model.tokens.schema import SCALE_INTERVALS, ScaleType
from musak_shared.elements import PITCHES_PER_OCTAVE

_GENERIC_THIRD_STEP: Final = 2
_GENERIC_FIFTH_STEP: Final = 4


def natural_triad(scale_type: ScaleType, degree: int) -> Chord:
    intervals = SCALE_INTERVALS[scale_type]
    scale_size = len(intervals)
    root_index = degree - 1
    root_semitone = intervals[root_index]
    third = (intervals[(root_index + _GENERIC_THIRD_STEP) % scale_size] - root_semitone) % PITCHES_PER_OCTAVE
    fifth = (intervals[(root_index + _GENERIC_FIFTH_STEP) % scale_size] - root_semitone) % PITCHES_PER_OCTAVE
    return Chord(root_degree=degree, root_accidental=0, quality=TRIAD_QUALITY_BY_INTERVALS[(third, fifth)])
