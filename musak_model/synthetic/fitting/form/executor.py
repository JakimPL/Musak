import logging
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Final

from musak_model.n_grams.profile.chord.schema import ChordDecodeSpec
from musak_model.processing.progress import progress
from musak_model.processing.workers import process_pool_context
from musak_model.synthetic.fitting.form.cadence import CadenceDetectionConfig
from musak_model.synthetic.fitting.form.repetition import RepetitionConfig
from musak_model.synthetic.fitting.form.store import FormWorkStore
from musak_model.synthetic.fitting.form.worker import FormBatchResult, FormBatchTask, process_form_batch_task
from musak_model.tokens.config import TokenizationConfig

_LOGGER = logging.getLogger(__name__)
_PROGRESS_DESCRIPTION: Final = "Counting form statistics batches"


def form_batch_tasks(
    encoded_jsonl_path: Path,
    *,
    tokenization_config: TokenizationConfig,
    chord_decode: ChordDecodeSpec,
    figure_min_n: int,
    figure_max_n: int,
    cadence_config: CadenceDetectionConfig,
    repetition_config: RepetitionConfig,
    batch_size: int,
    completed_batches: set[int],
) -> Iterator[FormBatchTask]:
    with encoded_jsonl_path.open("r", encoding="utf-8") as file:
        batch_index = 0
        sample_start_index = 0
        encoded_lines: list[str] = []
        for line in file:
            if line.strip() == "":
                continue

            encoded_lines.append(line)
            if len(encoded_lines) == batch_size:
                if batch_index not in completed_batches:
                    yield _form_batch_task(
                        batch_index=batch_index,
                        sample_start_index=sample_start_index,
                        encoded_lines=encoded_lines,
                        tokenization_config=tokenization_config,
                        chord_decode=chord_decode,
                        figure_min_n=figure_min_n,
                        figure_max_n=figure_max_n,
                        cadence_config=cadence_config,
                        repetition_config=repetition_config,
                    )

                batch_index += 1
                sample_start_index += len(encoded_lines)
                encoded_lines.clear()

        if encoded_lines and batch_index not in completed_batches:
            yield _form_batch_task(
                batch_index=batch_index,
                sample_start_index=sample_start_index,
                encoded_lines=encoded_lines,
                tokenization_config=tokenization_config,
                chord_decode=chord_decode,
                figure_min_n=figure_min_n,
                figure_max_n=figure_max_n,
                cadence_config=cadence_config,
                repetition_config=repetition_config,
            )


def process_missing_form_batches(
    store: FormWorkStore,
    *,
    encoded_jsonl_path: Path,
    tokenization_config: TokenizationConfig,
    chord_decode: ChordDecodeSpec,
    figure_min_n: int,
    figure_max_n: int,
    cadence_config: CadenceDetectionConfig,
    repetition_config: RepetitionConfig,
    batch_size: int,
    workers: int,
    show_progress: bool,
) -> None:
    _LOGGER.info(
        "Processing form statistics batches: completed=%s batch_size=%s workers=%s",
        len(store.completed_batch_indexes()),
        batch_size,
        workers,
    )
    tasks = form_batch_tasks(
        encoded_jsonl_path,
        tokenization_config=tokenization_config,
        chord_decode=chord_decode,
        figure_min_n=figure_min_n,
        figure_max_n=figure_max_n,
        cadence_config=cadence_config,
        repetition_config=repetition_config,
        batch_size=batch_size,
        completed_batches=store.completed_batch_indexes(),
    )
    if workers == 1:
        for task in progress(tasks, description=_PROGRESS_DESCRIPTION, unit="batch", enabled=show_progress):
            _commit(store, process_form_batch_task(task))
        return

    _process_in_parallel(store, tasks, workers=workers)


def _process_in_parallel(store: FormWorkStore, tasks: Iterator[FormBatchTask], *, workers: int) -> None:
    with ProcessPoolExecutor(max_workers=workers, mp_context=process_pool_context()) as executor:
        pending: set[Future[FormBatchResult]] = set()
        task_iterator = iter(tasks)
        _submit_pending(executor, pending, task_iterator, workers=workers)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                _commit(store, future.result())

            _submit_pending(
                executor,
                pending,
                task_iterator,
                workers=workers,
            )


def _submit_pending(
    executor: ProcessPoolExecutor,
    pending: set[Future[FormBatchResult]],
    tasks: Iterator[FormBatchTask],
    *,
    workers: int,
) -> None:
    while len(pending) < workers * 2:
        try:
            task = next(tasks)
        except StopIteration:
            return

        pending.add(executor.submit(process_form_batch_task, task))


def _commit(store: FormWorkStore, result: FormBatchResult) -> None:
    store.commit_batch(
        result.statistics,
        batch_index=result.batch_index,
        sample_start_index=result.sample_start_index,
        sample_count=result.encoded_sample_count,
    )


def _form_batch_task(
    *,
    batch_index: int,
    sample_start_index: int,
    encoded_lines: list[str],
    tokenization_config: TokenizationConfig,
    chord_decode: ChordDecodeSpec,
    figure_min_n: int,
    figure_max_n: int,
    cadence_config: CadenceDetectionConfig,
    repetition_config: RepetitionConfig,
) -> FormBatchTask:
    return FormBatchTask(
        batch_index=batch_index,
        sample_start_index=sample_start_index,
        encoded_lines=tuple(encoded_lines),
        tokenization_config=tokenization_config,
        chord_decode=chord_decode,
        figure_min_n=figure_min_n,
        figure_max_n=figure_max_n,
        cadence_config=cadence_config,
        repetition_config=repetition_config,
    )
