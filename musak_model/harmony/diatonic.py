from musak_model.harmony.schema import TRIAD_QUALITY_BY_INTERVALS, Chord
from musak_model.tokens.schema import MIN_DEGREE, SCALE_INTERVALS, ScaleType
from musak_shared.elements import PITCHES_PER_OCTAVE


def diatonic_triad(
    *,
    scale_type: ScaleType,
    root_degree: int,
) -> Chord:
    intervals = SCALE_INTERVALS[scale_type]
    scale_size = len(intervals)
    if not MIN_DEGREE <= root_degree <= scale_size:
        raise ValueError(f"root_degree must be in [{MIN_DEGREE}, {scale_size}], got {root_degree}")

    root_index = root_degree - 1
    root_semitone = intervals[root_index]
    third = (intervals[(root_index + 2) % scale_size] - root_semitone) % PITCHES_PER_OCTAVE
    fifth = (intervals[(root_index + 4) % scale_size] - root_semitone) % PITCHES_PER_OCTAVE
    return Chord(
        root_degree=root_degree,
        root_accidental=0,
        quality=TRIAD_QUALITY_BY_INTERVALS[(third, fifth)],
    )


def diatonic_triads(scale_type: ScaleType) -> tuple[Chord, ...]:
    scale_size = len(SCALE_INTERVALS[scale_type])
    return tuple(
        diatonic_triad(scale_type=scale_type, root_degree=degree) for degree in range(MIN_DEGREE, scale_size + 1)
    )
