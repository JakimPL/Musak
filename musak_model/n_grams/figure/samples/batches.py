from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from musak_model.n_grams.figure.samples.merge import merge_scale_counts
from musak_model.n_grams.figure.samples.schema import (
    EncodedExerciseFigureNGramCounts,
    FigureNGramCountsByScale,
)
from musak_model.n_grams.figure.samples.single import count_encoded_exercise_figure_ngrams
from musak_model.processing.progress import progress
from musak_model.processing.workers import process_pool_context
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


@dataclass(frozen=True)
class FigureBatchTask:
    index: int
    sample_start_index: int
    samples: tuple[EncodedExercise, ...]
    duration_vocabulary: DurationVocabulary
    token_vocabulary: TokenVocabulary
    min_n: int
    max_n: int


@dataclass(frozen=True)
class FigureBatchResult:
    index: int
    counts_by_scale: FigureNGramCountsByScale
    counts_by_sample: tuple[EncodedExerciseFigureNGramCounts, ...]


def run_figure_batch_tasks(
    tasks: tuple[FigureBatchTask, ...],
    *,
    workers: int,
    show_progress: bool,
    progress_description: str,
) -> tuple[FigureBatchResult, ...]:
    if not tasks:
        return ()

    if workers == 1:
        return _run_figure_batch_tasks_serially(
            tasks,
            show_progress=show_progress,
            progress_description=progress_description,
        )

    return _run_figure_batch_tasks_in_parallel(
        tasks,
        workers=workers,
        show_progress=show_progress,
        progress_description=progress_description,
    )


def process_figure_batch_task(task: FigureBatchTask) -> FigureBatchResult:
    counts_by_scale: FigureNGramCountsByScale = {}
    counts_by_sample: list[EncodedExerciseFigureNGramCounts] = []
    for sample_offset, sample in enumerate(task.samples):
        sample_counts = count_encoded_exercise_figure_ngrams(
            sample,
            duration_vocabulary=task.duration_vocabulary,
            token_vocabulary=task.token_vocabulary,
            min_n=task.min_n,
            max_n=task.max_n,
        )
        merge_scale_counts(
            counts_by_scale,
            scale_type=sample.scale_type,
            sample_counts=sample_counts,
        )
        counts_by_sample.append(
            EncodedExerciseFigureNGramCounts(
                sample_index=task.sample_start_index + sample_offset,
                scale_type=sample.scale_type,
                counts_by_hand=sample_counts,
            )
        )

    return FigureBatchResult(
        index=task.index,
        counts_by_scale=counts_by_scale,
        counts_by_sample=tuple(counts_by_sample),
    )


def figure_batch_tasks(
    samples: Sequence[EncodedExercise],
    *,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    min_n: int,
    max_n: int,
    batch_size: int,
) -> tuple[FigureBatchTask, ...]:
    sample_tuple = tuple(samples)
    return tuple(
        FigureBatchTask(
            index=index,
            sample_start_index=start_index,
            samples=sample_tuple[start_index : start_index + batch_size],
            duration_vocabulary=duration_vocabulary,
            token_vocabulary=token_vocabulary,
            min_n=min_n,
            max_n=max_n,
        )
        for index, start_index in enumerate(range(0, len(sample_tuple), batch_size))
    )


def _run_figure_batch_tasks_serially(
    tasks: tuple[FigureBatchTask, ...],
    *,
    show_progress: bool,
    progress_description: str,
) -> tuple[FigureBatchResult, ...]:
    results: list[FigureBatchResult] = []
    for task in progress(tasks, description=progress_description, unit="batch", enabled=show_progress):
        results.append(process_figure_batch_task(task))

    return tuple(results)


def _run_figure_batch_tasks_in_parallel(
    tasks: tuple[FigureBatchTask, ...],
    *,
    workers: int,
    show_progress: bool,
    progress_description: str,
) -> tuple[FigureBatchResult, ...]:
    ordered_results: list[FigureBatchResult | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=workers, mp_context=process_pool_context()) as executor:
        futures: dict[Future[FigureBatchResult], int] = {
            executor.submit(process_figure_batch_task, task): task.index for task in tasks
        }
        completed_futures = as_completed(futures)
        for future in progress(
            completed_futures,
            total=len(futures),
            description=progress_description,
            unit="batch",
            enabled=show_progress,
        ):
            result = future.result()
            ordered_results[result.index] = result
            del futures[future]

    return tuple(result for result in ordered_results if result is not None)
