from collections import Counter
from collections.abc import Mapping, Sequence

from musak_model.analysis.n_grams.figure.parser import HandOnsetRun
from musak_model.analysis.n_grams.figure.schema import FigureNGram
from musak_model.analysis.n_grams.figure.signature import (
    figure_signature_to_ngram,
    iter_figure_signatures_from_run,
)
from musak_model.tokens.schema import Hand

type FigureNGramCounter = Counter[FigureNGram]
type FigureNGramCountsByN = dict[int, FigureNGramCounter]
type FigureNGramCountsByHand = dict[Hand, FigureNGramCountsByN]


def count_figure_ngrams(
    runs: Sequence[HandOnsetRun],
    *,
    min_n: int,
    max_n: int,
    scale_size: int,
) -> FigureNGramCountsByN:
    if min_n <= 0:
        raise ValueError("min_n must be positive")

    if max_n < min_n:
        raise ValueError("max_n must be greater than or equal to min_n")

    counts_by_n: FigureNGramCountsByN = {n: Counter() for n in range(min_n, max_n + 1)}
    for run in runs:
        for n, signature in iter_figure_signatures_from_run(
            run,
            min_n=min_n,
            max_n=max_n,
            scale_size=scale_size,
        ):
            counts_by_n[n][figure_signature_to_ngram(signature)] += 1

    return counts_by_n


def count_hand_figure_ngrams(
    runs_by_hand: Mapping[Hand, Sequence[HandOnsetRun]],
    *,
    min_n: int,
    max_n: int,
    scale_size: int,
) -> FigureNGramCountsByHand:
    return {
        hand: count_figure_ngrams(
            runs,
            min_n=min_n,
            max_n=max_n,
            scale_size=scale_size,
        )
        for hand, runs in runs_by_hand.items()
    }
