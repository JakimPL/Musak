from collections import Counter
from collections.abc import Sequence
from copy import deepcopy

from musak_model.analysis.n_grams.builder import scale_size_for_type
from musak_model.analysis.n_grams.counter import (
    FigureNGramCountsByHand,
    count_hand_figure_ngrams,
)
from musak_model.analysis.n_grams.parser import extract_hand_onset_runs
from musak_model.processing.progress import progress
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise

type FigureNGramCountsByScale = dict[ScaleType, FigureNGramCountsByHand]


def count_encoded_exercise_figure_ngrams(
    sample: EncodedExercise,
    *,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    min_n: int,
    max_n: int,
) -> FigureNGramCountsByHand:
    tokens = token_vocabulary.decode(sample.token_ids)
    runs_by_hand = extract_hand_onset_runs(
        tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=sample.time_numerator,
        time_denominator=sample.time_denominator,
    )
    return count_hand_figure_ngrams(
        runs_by_hand,
        min_n=min_n,
        max_n=max_n,
        scale_size=scale_size_for_type(sample.scale_type),
    )


def count_encoded_exercises_figure_ngrams(
    samples: Sequence[EncodedExercise],
    *,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    min_n: int,
    max_n: int,
    show_progress: bool = False,
) -> FigureNGramCountsByScale:
    counts_by_scale: FigureNGramCountsByScale = {}
    for sample in progress(
        samples,
        description="Counting figure n-grams",
        unit="sample",
        enabled=show_progress,
        total=len(samples),
    ):
        sample_counts = count_encoded_exercise_figure_ngrams(
            sample,
            duration_vocabulary=duration_vocabulary,
            token_vocabulary=token_vocabulary,
            min_n=min_n,
            max_n=max_n,
        )
        _merge_scale_counts(
            counts_by_scale,
            scale_type=sample.scale_type,
            sample_counts=sample_counts,
        )

    return counts_by_scale


def _merge_scale_counts(
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
