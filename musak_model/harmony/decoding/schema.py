from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from musak_model.data.schema import Segment
from musak_model.harmony.schema import Chord
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.tokens.duration import DurationVocabulary


@dataclass(frozen=True)
class ChordWindow:
    start: Fraction
    end: Fraction
    chord: Chord


class ChordDecoder(Protocol):
    def decode(
        self,
        segment: Segment,
        *,
        duration_vocabulary: DurationVocabulary,
        vocabulary: ChordVocabularyConfig,
    ) -> tuple[ChordWindow, ...]: ...
