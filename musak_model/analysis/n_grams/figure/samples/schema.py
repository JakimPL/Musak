from dataclasses import dataclass

from musak_model.analysis.n_grams.figure.counter import FigureNGramCountsByHand
from musak_model.tokens.schema import ScaleType

type FigureNGramCountsByScale = dict[ScaleType, FigureNGramCountsByHand]


@dataclass(frozen=True)
class EncodedExerciseFigureNGramCounts:
    sample_index: int
    scale_type: ScaleType
    counts_by_hand: FigureNGramCountsByHand


@dataclass(frozen=True)
class EncodedFigureNGramCounts:
    counts_by_scale: FigureNGramCountsByScale
    counts_by_sample: tuple[EncodedExerciseFigureNGramCounts, ...]
