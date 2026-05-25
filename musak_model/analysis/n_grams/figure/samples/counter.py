from collections.abc import Sequence

from musak_model.analysis.n_grams.figure.samples.batches import figure_batch_tasks, run_figure_batch_tasks
from musak_model.analysis.n_grams.figure.samples.merge import merge_batch_counts
from musak_model.analysis.n_grams.figure.samples.schema import (
    EncodedExerciseFigureNGramCounts,
    EncodedFigureNGramCounts,
    FigureNGramCountsByScale,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


def count_encoded_exercises_figure_ngrams(
    samples: Sequence[EncodedExercise],
    *,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    min_n: int,
    max_n: int,
    workers: int,
    batch_size: int,
    show_progress: bool = False,
) -> FigureNGramCountsByScale:
    return count_encoded_exercises_figure_ngrams_with_samples(
        samples,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=min_n,
        max_n=max_n,
        workers=workers,
        batch_size=batch_size,
        show_progress=show_progress,
    ).counts_by_scale


def count_encoded_exercises_figure_ngrams_with_samples(
    samples: Sequence[EncodedExercise],
    *,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    min_n: int,
    max_n: int,
    workers: int,
    batch_size: int,
    show_progress: bool = False,
) -> EncodedFigureNGramCounts:
    if workers <= 0:
        raise ValueError("workers must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    tasks = figure_batch_tasks(
        samples,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=min_n,
        max_n=max_n,
        batch_size=batch_size,
    )
    batch_results = run_figure_batch_tasks(
        tasks,
        workers=workers,
        show_progress=show_progress,
    )
    counts_by_scale: FigureNGramCountsByScale = {}
    counts_by_sample: list[EncodedExerciseFigureNGramCounts] = []
    for batch_result in batch_results:
        merge_batch_counts(counts_by_scale, batch_result.counts_by_scale)
        counts_by_sample.extend(batch_result.counts_by_sample)

    return EncodedFigureNGramCounts(
        counts_by_scale=counts_by_scale,
        counts_by_sample=tuple(sorted(counts_by_sample, key=lambda sample_counts: sample_counts.sample_index)),
    )


def count_encoded_exercises_figure_n_grams(
    samples: Sequence[EncodedExercise],
    *,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    min_n: int,
    max_n: int,
    workers: int,
    batch_size: int,
    show_progress: bool = False,
) -> FigureNGramCountsByScale:
    return count_encoded_exercises_figure_ngrams(
        samples,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=min_n,
        max_n=max_n,
        workers=workers,
        batch_size=batch_size,
        show_progress=show_progress,
    )
