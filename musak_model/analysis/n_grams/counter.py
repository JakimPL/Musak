from collections import Counter
from collections.abc import Mapping, Sequence

from musak_model.analysis.n_grams.builder import build_figure_ngrams_from_run
from musak_model.analysis.n_grams.figure import FigureNGram
from musak_model.analysis.n_grams.parser import HandOnsetRun
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
        for n in counts_by_n:
            counts_by_n[n].update(
                build_figure_ngrams_from_run(
                    run,
                    n=n,
                    scale_size=scale_size,
                )
            )

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
