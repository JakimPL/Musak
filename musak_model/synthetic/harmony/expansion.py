from dataclasses import dataclass
from typing import Final

from musak_model.synthetic.harmony.schema import Chord
from musak_model.synthetic.harmony.vocabulary import ChordVocabularyConfig
from musak_model.tokens.schema import MAX_ACCIDENTAL, MIN_ACCIDENTAL, SCALE_INTERVALS, ScaleType
from musak_shared.elements import PITCHES_PER_OCTAVE

_GENERIC_THIRD_STEP: Final[int] = 2


@dataclass(frozen=True)
class ChordTone:
    degree: int
    accidental: int


class UnspellableChordError(ValueError):
    """Raised when a chord member needs an accidental outside the notatable range."""


def expand_chord_to_tones(
    chord: Chord,
    *,
    scale_type: ScaleType,
    vocabulary: ChordVocabularyConfig,
) -> tuple[ChordTone, ...]:
    intervals = SCALE_INTERVALS[scale_type]
    scale_size = len(intervals)
    if chord.root_degree > scale_size:
        raise ValueError(
            f"chord root degree {chord.root_degree} exceeds the {scale_size}-degree {scale_type.value} scale"
        )

    quality_definition = vocabulary.quality_definition(chord.quality)
    extension_definition = vocabulary.extension_definition(chord.extension)
    triad_interval_count = len(quality_definition.intervals)
    root_index = chord.root_degree - 1
    root_semitone = intervals[root_index] + chord.root_accidental

    tones: list[ChordTone] = []
    for member in range(extension_definition.members):
        generic_index = (root_index + _GENERIC_THIRD_STEP * member) % scale_size
        natural_semitone = intervals[generic_index]
        if member < triad_interval_count:
            desired_semitone = root_semitone + quality_definition.intervals[member]
            accidental = _signed_pitch_class_residue(desired_semitone - natural_semitone)
        else:
            accidental = 0

        accidental += extension_definition.alterations.get(member, 0)
        if not MIN_ACCIDENTAL <= accidental <= MAX_ACCIDENTAL:
            raise UnspellableChordError(
                f"chord member {member} of {chord!r} requires accidental {accidental} "
                f"outside the notatable range [{MIN_ACCIDENTAL}, {MAX_ACCIDENTAL}]"
            )

        tones.append(ChordTone(degree=generic_index + 1, accidental=accidental))

    return tuple(tones)


def _signed_pitch_class_residue(value: int) -> int:
    residue = value % PITCHES_PER_OCTAVE
    if residue > PITCHES_PER_OCTAVE // 2:
        return residue - PITCHES_PER_OCTAVE

    return residue
