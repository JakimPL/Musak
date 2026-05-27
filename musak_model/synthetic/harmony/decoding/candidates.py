from collections.abc import Iterator
from dataclasses import dataclass

from musak_model.n_grams.figure.builder import scale_size_for_type
from musak_model.synthetic.harmony.expansion import UnspellableChordError, expand_chord_to_tones
from musak_model.synthetic.harmony.schema import Chord
from musak_model.synthetic.harmony.vocabulary import ChordVocabularyConfig
from musak_model.tokens.pitch import degree_pitch_class
from musak_model.tokens.schema import MIN_DEGREE, ScaleType


@dataclass(frozen=True)
class Candidate:
    chord: Chord
    pitch_classes: frozenset[int]


def spellable_candidates(vocabulary: ChordVocabularyConfig, *, scale_type: ScaleType) -> tuple[Candidate, ...]:
    scale_size = scale_size_for_type(scale_type)
    candidates: list[Candidate] = []
    for chord in _candidate_chords(vocabulary, scale_size=scale_size):
        try:
            tones = expand_chord_to_tones(chord, scale_type=scale_type, vocabulary=vocabulary)
        except UnspellableChordError:
            continue

        pitch_classes = frozenset(
            degree_pitch_class(tone.degree, tone.accidental, scale_type=scale_type) for tone in tones
        )
        candidates.append(Candidate(chord=chord, pitch_classes=pitch_classes))

    return tuple(candidates)


def _candidate_chords(vocabulary: ChordVocabularyConfig, *, scale_size: int) -> Iterator[Chord]:
    for root_degree in range(MIN_DEGREE, scale_size + 1):
        for quality in vocabulary.enabled_qualities():
            for extension in vocabulary.enabled_extensions():
                yield Chord(root_degree=root_degree, root_accidental=0, quality=quality, extension=extension)
