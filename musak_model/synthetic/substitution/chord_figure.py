from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.harmony.schema import Chord
from musak_model.tokens.schema import Hand, ScaleType

type FigureByChordKey = tuple[ScaleType, Hand, int, Chord]


@dataclass(frozen=True)
class FigureByChordModel:
    log_probabilities: Mapping[FigureByChordKey, Mapping[FigureNGram, float]] = field(default_factory=dict)

    def table(
        self,
        *,
        scale_type: ScaleType,
        hand: Hand,
        figure_length: int,
        chord: Chord | None,
    ) -> Mapping[FigureNGram, float] | None:
        if chord is None:
            return None

        return self.log_probabilities.get((scale_type, hand, figure_length, chord))
