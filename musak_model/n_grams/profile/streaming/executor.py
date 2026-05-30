import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from time import perf_counter
from typing import Protocol, Self, cast

from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.profile.chord.schema import ChordDecodeSpec
from musak_model.n_grams.profile.streaming.schema import FigureBatchResult, FigureBatchTask
from musak_model.n_grams.profile.streaming.store import FigureWorkStore
from musak_model.n_grams.profile.streaming.tasks import figure_batch_tasks, figure_sample_batch_tasks
from musak_model.n_grams.profile.streaming.worker import process_figure_batch_task
from musak_model.processing.progress import progress
from musak_model.processing.workers import process_pool_context
from musak_model.tokens.config import TokenizationConfig
from musak_model.training.ingestion.schema import EncodedExercise

_LOGGER = logging.getLogger(__name__)


class ProgressBar(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> object: ...

    def update(self, value: int) -> object: ...


class NullProgressBar:
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        _ = exc_type, exc_value, traceback

    def update(self, value: int) -> None:
        _ = value


def process_missing_batches(
    store: FigureWorkStore,
    *,
    encoded_jsonl_path: Path,
    tokenization_config: TokenizationConfig,
    config: NGramAnalysisConfig,
    chord_decode: ChordDecodeSpec | None,
    show_progress: bool,
) -> None:
    completed_batches = store.completed_batch_indexes()
    _LOGGER.info(
        "Preparing figure n-gram batches: encoded_jsonl=%s completed_batches=%s batch_size=%s workers=%s",
        encoded_jsonl_path,
        len(completed_batches),
        config.execution.batch_size,
        config.execution.workers,
    )
    tasks = figure_batch_tasks(
        encoded_jsonl_path,
        tokenization_config=tokenization_config,
        min_n=config.figure.min_n,
        max_n=config.figure.max_n,
        rhythm_min_n=config.rhythm.min_n,
        rhythm_max_n=config.rhythm.max_n,
        grid_alignment_denominators=config.rhythm.grid_alignment_denominators,
        strong_beat_offsets=config.rhythm.strong_beat_offsets,
        register_arch_basis_count=config.register.arch_basis_count,
        chord_decode=chord_decode,
        batch_size=config.execution.batch_size,
        completed_batches=completed_batches,
    )
    process_figure_batch_tasks(
        store,
        tasks,
        workers=config.execution.workers,
        show_progress=show_progress,
        progress_description="Counting figure n-gram batches",
    )


def process_missing_sample_batches(
    store: FigureWorkStore,
    *,
    samples: Iterable[EncodedExercise],
    tokenization_config: TokenizationConfig,
    config: NGramAnalysisConfig,
    show_progress: bool,
    progress_description: str,
) -> None:
    completed_batches = store.completed_batch_indexes()
    _LOGGER.info(
        "Preparing in-memory figure n-gram batches: completed_batches=%s batch_size=%s workers=%s",
        len(completed_batches),
        config.execution.batch_size,
        config.execution.workers,
    )
    tasks = figure_sample_batch_tasks(
        samples,
        tokenization_config=tokenization_config,
        min_n=config.figure.min_n,
        max_n=config.figure.max_n,
        rhythm_min_n=config.rhythm.min_n,
        rhythm_max_n=config.rhythm.max_n,
        grid_alignment_denominators=config.rhythm.grid_alignment_denominators,
        strong_beat_offsets=config.rhythm.strong_beat_offsets,
        register_arch_basis_count=config.register.arch_basis_count,
        batch_size=config.execution.batch_size,
        completed_batches=completed_batches,
    )
    process_figure_batch_tasks(
        store,
        tasks,
        workers=config.execution.workers,
        show_progress=show_progress,
        progress_description=progress_description,
    )


def process_figure_batch_tasks(
    store: FigureWorkStore,
    tasks: Iterator[FigureBatchTask],
    *,
    workers: int,
    show_progress: bool,
    progress_description: str,
) -> None:
    if workers == 1:
        started_at = perf_counter()
        for task in progress(tasks, description=progress_description, unit="batch", enabled=show_progress):
            store.commit_batch(process_figure_batch_task(task))
        _LOGGER.info("Finished serial figure n-gram batches in %.1fs", perf_counter() - started_at)
        return

    started_at = perf_counter()
    _process_missing_batches_in_parallel(
        store,
        tasks,
        workers=workers,
        show_progress=show_progress,
        progress_description=progress_description,
    )
    _LOGGER.info("Finished parallel figure n-gram batches in %.1fs", perf_counter() - started_at)


def _process_missing_batches_in_parallel(
    store: FigureWorkStore,
    tasks: Iterator[FigureBatchTask],
    *,
    workers: int,
    show_progress: bool,
    progress_description: str,
) -> None:
    _LOGGER.info("Running parallel figure n-gram batches: workers=%s", workers)
    with ProcessPoolExecutor(max_workers=workers, mp_context=process_pool_context()) as executor:
        pending: dict[Future[FigureBatchResult], int] = {}
        task_iterator = iter(tasks)
        _submit_pending_tasks(executor, pending, task_iterator, workers=workers)
        completed_count = 0
        with _parallel_progress(progress_description, enabled=show_progress) as progress_bar:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    store.commit_batch(result)
                    completed_count += 1
                    progress_bar.update(1)
                    del pending[future]
                _submit_pending_tasks(executor, pending, task_iterator, workers=workers)
        _LOGGER.info("Completed %s figure n-gram batch(es)", completed_count)


def _parallel_progress(
    description: str,
    *,
    enabled: bool,
) -> ProgressBar:
    if not enabled:
        return NullProgressBar()

    from tqdm.auto import tqdm

    return cast(ProgressBar, tqdm(desc=description, unit="batch"))


def _submit_pending_tasks(
    executor: ProcessPoolExecutor,
    pending: dict[Future[FigureBatchResult], int],
    tasks: Iterator[FigureBatchTask],
    *,
    workers: int,
) -> None:
    max_pending = workers * 2
    while len(pending) < max_pending:
        try:
            task = next(tasks)
        except StopIteration:
            return

        pending[executor.submit(process_figure_batch_task, task)] = task.batch_index
