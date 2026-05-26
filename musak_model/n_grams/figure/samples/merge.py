from collections import Counter
from copy import deepcopy

from musak_model.n_grams.figure.counter import FigureNGramCountsByHand
from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.tokens.schema import Hand, ScaleType


def merge_batch_counts(
    counts_by_scale: FigureNGramCountsByScale,
    batch_counts: FigureNGramCountsByScale,
) -> None:
    for scale_type, counts_by_hand in batch_counts.items():
        if scale_type not in counts_by_scale:
            counts_by_scale[scale_type] = deepcopy(counts_by_hand)
            continue

        scale_counts = counts_by_scale[scale_type]
        for hand in Hand:
            hand_counts = counts_by_hand.get(hand, {})
            for n, figure_counts in hand_counts.items():
                scale_counts.setdefault(hand, {}).setdefault(n, Counter()).update(figure_counts)


def merge_scale_counts(
    counts_by_scale: FigureNGramCountsByScale,
    *,
    scale_type: ScaleType,
    sample_counts: FigureNGramCountsByHand,
) -> None:
    if scale_type not in counts_by_scale:
        counts_by_scale[scale_type] = deepcopy(sample_counts)
        return

    scale_counts = counts_by_scale[scale_type]
    for hand in Hand:
        hand_counts = sample_counts.get(hand, {})
        for n, figure_counts in hand_counts.items():
            scale_counts.setdefault(hand, {}).setdefault(n, Counter()).update(figure_counts)
