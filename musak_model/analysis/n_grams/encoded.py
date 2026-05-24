from collections import Counter
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass

from musak_model.analysis.n_grams.builder import scale_size_for_type
from musak_model.analysis.n_grams.counter import (
    FigureNGramCountsByHand,
    count_hand_figure_ngrams,
)
from musak_model.analysis.n_grams.parser import extract_hand_onset_runs
from musak_model.processing.progress import progress
from musak_model.processing.workers import process_pool_context
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise

type FigureNGramCountsByScale = dict[ScaleType, FigureNGramCountsByHand]


@dataclass(frozen=True)
class _NGramBatchTask:
    index: int
    samples: tuple[EncodedExercise, ...]
    duration_vocabulary: DurationVocabulary
    token_vocabulary: TokenVocabulary
    min_n: int
    max_n: int


@dataclass(frozen=True)
class _NGramBatchResult:
    index: int
    counts_by_scale: FigureNGramCountsByScale


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
    workers: int,
    batch_size: int,
    show_progress: bool = False,
) -> FigureNGramCountsByScale:
    if workers <= 0:
        raise ValueError("workers must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    tasks = _ngram_batch_tasks(
        samples,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=min_n,
        max_n=max_n,
        batch_size=batch_size,
    )
    batch_results = run_ngram_batch_tasks(
        tasks,
        workers=workers,
        show_progress=show_progress,
    )
    counts_by_scale: FigureNGramCountsByScale = {}
    for batch_result in batch_results:
        _merge_batch_counts(counts_by_scale, batch_result.counts_by_scale)

    return counts_by_scale


def run_ngram_batch_tasks(
    tasks: tuple[_NGramBatchTask, ...],
    *,
    workers: int,
    show_progress: bool,
) -> tuple[_NGramBatchResult, ...]:
    if not tasks:
        return ()

    if workers == 1:
        return _run_ngram_batch_tasks_serially(tasks, show_progress=show_progress)

    return _run_ngram_batch_tasks_in_parallel(tasks, workers=workers, show_progress=show_progress)


def process_ngram_batch_task(task: _NGramBatchTask) -> _NGramBatchResult:
    counts_by_scale: FigureNGramCountsByScale = {}
    for sample in task.samples:
        sample_counts = count_encoded_exercise_figure_ngrams(
            sample,
            duration_vocabulary=task.duration_vocabulary,
            token_vocabulary=task.token_vocabulary,
            min_n=task.min_n,
            max_n=task.max_n,
        )
        _merge_scale_counts(
            counts_by_scale,
            scale_type=sample.scale_type,
            sample_counts=sample_counts,
        )

    return _NGramBatchResult(index=task.index, counts_by_scale=counts_by_scale)


def _run_ngram_batch_tasks_serially(
    tasks: tuple[_NGramBatchTask, ...],
    *,
    show_progress: bool,
) -> tuple[_NGramBatchResult, ...]:
    results: list[_NGramBatchResult] = []
    for task in progress(tasks, description="Counting figure n-gram batches", unit="batch", enabled=show_progress):
        results.append(process_ngram_batch_task(task))

    return tuple(results)


def _run_ngram_batch_tasks_in_parallel(
    tasks: tuple[_NGramBatchTask, ...],
    *,
    workers: int,
    show_progress: bool,
) -> tuple[_NGramBatchResult, ...]:
    ordered_results: list[_NGramBatchResult | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=workers, mp_context=process_pool_context()) as executor:
        futures: dict[Future[_NGramBatchResult], int] = {
            executor.submit(process_ngram_batch_task, task): task.index for task in tasks
        }
        completed_futures = as_completed(futures)
        for future in progress(
            completed_futures,
            total=len(futures),
            description="Counting figure n-gram batches",
            unit="batch",
            enabled=show_progress,
        ):
            result = future.result()
            ordered_results[result.index] = result
            del futures[future]

    return tuple(result for result in ordered_results if result is not None)


def _ngram_batch_tasks(
    samples: Sequence[EncodedExercise],
    *,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    min_n: int,
    max_n: int,
    batch_size: int,
) -> tuple[_NGramBatchTask, ...]:
    sample_tuple = tuple(samples)
    return tuple(
        _NGramBatchTask(
            index=index,
            samples=sample_tuple[start_index : start_index + batch_size],
            duration_vocabulary=duration_vocabulary,
            token_vocabulary=token_vocabulary,
            min_n=min_n,
            max_n=max_n,
        )
        for index, start_index in enumerate(range(0, len(sample_tuple), batch_size))
    )


def _merge_batch_counts(
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
