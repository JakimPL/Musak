from concurrent.futures import Future, ProcessPoolExecutor, as_completed

from musak_model.processing.parser import ParsedScoreArtifact
from musak_model.processing.profiler import NULL_PROCESSING_PROFILER
from musak_model.processing.progress import progress
from musak_model.processing.tokenizer.output import clear_tokenized_source_temp_files
from musak_model.processing.tokenizer.schema import (
    TokenizationBatchResult,
    TokenizationBatchTask,
    TokenizedSourceResult,
)
from musak_model.processing.tokenizer.source import tokenize_source
from musak_model.processing.workers import process_pool_context
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary


def run_tokenization_batch_tasks(
    tasks: tuple[TokenizationBatchTask, ...],
    *,
    workers: int,
    show_progress: bool,
) -> tuple[TokenizationBatchResult, ...]:
    if not tasks:
        return ()

    if workers == 1:
        return _run_tokenization_batch_tasks_serially(tasks, show_progress=show_progress)

    return _run_tokenization_batch_tasks_in_parallel(tasks, workers=workers, show_progress=show_progress)


def process_tokenization_batch_task(task: TokenizationBatchTask) -> TokenizationBatchResult:
    duration_vocabulary = DurationVocabulary(task.tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    source_results: list[TokenizedSourceResult] = []
    for artifact in task.artifacts:
        source_results.append(
            _process_tokenization_source(
                artifact,
                task=task,
                duration_vocabulary=duration_vocabulary,
                token_vocabulary=token_vocabulary,
            )
        )

    return TokenizationBatchResult(index=task.index, source_results=tuple(source_results))


def _process_tokenization_source(
    artifact: ParsedScoreArtifact,
    *,
    task: TokenizationBatchTask,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> TokenizedSourceResult:
    temp_source_root = task.temp_root / artifact.source_id_value
    temp_encoded_jsonl_path = temp_source_root / "data.jsonl"
    temp_encoded_manifest_path = temp_source_root / "encoded.csv"
    clear_tokenized_source_temp_files(
        encoded_jsonl_path=temp_encoded_jsonl_path,
        encoded_manifest_path=temp_encoded_manifest_path,
    )
    encoded_count, manifest_row_count, _ = tokenize_source(
        artifact,
        dataset_root=task.dataset_root,
        paths=task.paths,
        encoded_jsonl_path=temp_encoded_jsonl_path,
        manifest_encoded_jsonl_path=task.final_encoded_jsonl_path,
        encoded_manifest_path=temp_encoded_manifest_path,
        segmentation_config=task.segmentation_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        difficulty_labels=task.difficulty_labels,
        tokenization_processing_config=task.tokenization_processing_config,
        encoded_line_count=0,
        profiler=NULL_PROCESSING_PROFILER,
    )
    return TokenizedSourceResult(
        source_id_value=artifact.source_id_value,
        temp_encoded_jsonl_path=temp_encoded_jsonl_path,
        temp_encoded_manifest_path=temp_encoded_manifest_path,
        encoded_count=encoded_count,
        manifest_row_count=manifest_row_count,
    )


def _run_tokenization_batch_tasks_serially(
    tasks: tuple[TokenizationBatchTask, ...],
    *,
    show_progress: bool,
) -> tuple[TokenizationBatchResult, ...]:
    results: list[TokenizationBatchResult] = []
    for task in progress(tasks, description="Tokenizing batches", unit="batch", enabled=show_progress):
        results.append(process_tokenization_batch_task(task))

    return tuple(results)


def _run_tokenization_batch_tasks_in_parallel(
    tasks: tuple[TokenizationBatchTask, ...],
    *,
    workers: int,
    show_progress: bool,
) -> tuple[TokenizationBatchResult, ...]:
    ordered_results: list[TokenizationBatchResult | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=workers, mp_context=process_pool_context()) as executor:
        futures: dict[Future[TokenizationBatchResult], int] = {
            executor.submit(process_tokenization_batch_task, task): task.index for task in tasks
        }
        completed_futures = as_completed(futures)
        for future in progress(
            completed_futures,
            total=len(futures),
            description="Tokenizing batches",
            unit="batch",
            enabled=show_progress,
        ):
            result = future.result()
            ordered_results[result.index] = result
            del futures[future]

    return tuple(_filled_results(ordered_results))


def _filled_results(
    results: list[TokenizationBatchResult | None],
) -> list[TokenizationBatchResult]:
    return [result for result in results if result is not None]
