import logging
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

from musak_model.data.config import SegmentationConfig
from musak_model.processing.config import TokenizationProcessingConfig
from musak_model.processing.io import write_json_model
from musak_model.processing.manifest import EncodedManifestField, read_encoded_manifest
from musak_model.processing.parser import ParsedScoreArtifact, load_parsed_score_artifacts
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.profiler import NULL_PROCESSING_PROFILER, ProcessingProfilerProtocol
from musak_model.processing.progress import progress
from musak_model.processing.snapshot import TokenizerSnapshot, build_tokenizer_snapshot
from musak_model.processing.tokenizer.difficulty import log_difficulty_label_stats
from musak_model.processing.tokenizer.finalizer import merge_tokenized_source_result
from musak_model.processing.tokenizer.output import clear_tokenization_temp_root
from musak_model.processing.tokenizer.plan import batched_tokenization_sources, missing_tokenization_sources
from musak_model.processing.tokenizer.resume import (
    TokenizationOutputPaths,
    complete_outputs_exist,
    prepare_resume_state,
)
from musak_model.processing.tokenizer.schema import TokenizationBatchTask, TokenizeDatasetResult
from musak_model.processing.tokenizer.source import tokenize_source
from musak_model.processing.tokenizer.state import (
    TokenizationResumeState,
    append_source_completed_event,
    tokenization_state_key,
)
from musak_model.processing.tokenizer.worker import run_tokenization_batch_tasks
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary

_LOGGER = logging.getLogger(__name__)


def tokenize_dataset(
    dataset_root: Path,
    *,
    processed_root: Path,
    segmentation_config: SegmentationConfig,
    tokenization_config: TokenizationConfig,
    tokenization_processing_config: TokenizationProcessingConfig,
    difficulty_labels: dict[str, int | None] | None,
    overwrite: bool,
    show_progress: bool,
    profiler: ProcessingProfilerProtocol = NULL_PROCESSING_PROFILER,
) -> TokenizeDatasetResult:
    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    parsed_scores = load_parsed_score_artifacts(dataset_root, processed_root=processed_root)
    duration_vocabulary, token_vocabulary, snapshot = _tokenizer_components(tokenization_config)
    return tokenize_parsed_scores(
        parsed_scores,
        dataset_root=dataset_root,
        paths=paths,
        snapshot=snapshot,
        tokenization_config=tokenization_config,
        segmentation_config=segmentation_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        difficulty_labels=difficulty_labels,
        tokenization_processing_config=tokenization_processing_config,
        overwrite=overwrite,
        show_progress=show_progress,
        profiler=profiler,
    )


def tokenize_parsed_scores(
    parsed_scores: Iterable[ParsedScoreArtifact],
    *,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    snapshot: TokenizerSnapshot,
    tokenization_config: TokenizationConfig,
    segmentation_config: SegmentationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    difficulty_labels: dict[str, int | None] | None,
    tokenization_processing_config: TokenizationProcessingConfig,
    overwrite: bool,
    show_progress: bool,
    profiler: ProcessingProfilerProtocol = NULL_PROCESSING_PROFILER,
) -> TokenizeDatasetResult:
    parsed_scores = tuple(parsed_scores)
    output_paths = TokenizationOutputPaths.from_paths(paths, snapshot=snapshot)
    _LOGGER.info("Preparing tokenization resume state: %s", output_paths.state_path)
    started_at = perf_counter()
    state_key = tokenization_state_key(
        snapshot=snapshot,
        segmentation_config=segmentation_config,
        tokenization_processing_config=tokenization_processing_config,
        difficulty_labels=difficulty_labels,
    )
    resume_state = prepare_resume_state(
        output_paths,
        state_key=state_key,
        overwrite=overwrite,
        parsed_scores=parsed_scores,
        profiler=profiler,
    )
    _LOGGER.info("Prepared tokenization resume state in %.1fs", perf_counter() - started_at)
    reusable_result = _reusable_result(
        parsed_scores=parsed_scores,
        paths=paths,
        output_paths=output_paths,
        resume_state=resume_state,
        tokenization_processing_config=tokenization_processing_config,
    )
    if reusable_result is not None:
        return reusable_result

    _LOGGER.info("Writing tokenizer snapshot: %s", output_paths.tokenizer_snapshot_path)
    _write_tokenizer_snapshot(output_paths.tokenizer_snapshot_path, snapshot=snapshot, profiler=profiler)
    return _tokenize_missing_sources(
        parsed_scores,
        dataset_root=dataset_root,
        paths=paths,
        output_paths=output_paths,
        state_key=state_key,
        resume_state=resume_state,
        tokenization_config=tokenization_config,
        segmentation_config=segmentation_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        difficulty_labels=difficulty_labels,
        tokenization_processing_config=tokenization_processing_config,
        show_progress=show_progress,
        profiler=profiler,
    )


def _tokenizer_components(
    tokenization_config: TokenizationConfig,
) -> tuple[DurationVocabulary, TokenVocabulary, TokenizerSnapshot]:
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    return duration_vocabulary, token_vocabulary, snapshot


def _reusable_result(
    *,
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    paths: ProcessedDatasetPaths,
    output_paths: TokenizationOutputPaths,
    resume_state: TokenizationResumeState,
    tokenization_processing_config: TokenizationProcessingConfig,
) -> TokenizeDatasetResult | None:
    if not complete_outputs_exist(parsed_scores=parsed_scores, output_paths=output_paths, resume_state=resume_state):
        return None

    encoded_rows = read_encoded_manifest(output_paths.encoded_manifest_path)
    encoded_count = sum(1 for row in encoded_rows if row[EncodedManifestField.ENCODED_LINE] != "")
    _LOGGER.info("Reusing complete encoded artifacts: %s", output_paths.encoded_manifest_path)
    return _result(
        paths=paths,
        output_paths=output_paths,
        encoded_count=encoded_count,
        segment_count=len(encoded_rows),
        tokenization_processing_config=tokenization_processing_config,
    )


def _write_tokenizer_snapshot(
    tokenizer_snapshot_path: Path,
    *,
    snapshot: TokenizerSnapshot,
    profiler: ProcessingProfilerProtocol,
) -> None:
    with profiler.measure("write_tokenizer_snapshot"):
        write_json_model(snapshot, tokenizer_snapshot_path, overwrite=True)


def _tokenize_missing_sources(
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    *,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    output_paths: TokenizationOutputPaths,
    state_key: str,
    resume_state: TokenizationResumeState,
    tokenization_config: TokenizationConfig,
    segmentation_config: SegmentationConfig,
    tokenization_processing_config: TokenizationProcessingConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    difficulty_labels: dict[str, int | None] | None,
    show_progress: bool,
    profiler: ProcessingProfilerProtocol,
) -> TokenizeDatasetResult:
    _LOGGER.info("Encoding %s parsed score(s)", len(parsed_scores))
    log_difficulty_label_stats(parsed_scores, dataset_root=dataset_root, difficulty_labels=difficulty_labels)
    completed_source_ids = set(resume_state.completed_source_ids)
    remaining_scores = missing_tokenization_sources(parsed_scores, completed_source_ids=completed_source_ids)
    _LOGGER.info(
        "Tokenization resume state: completed_sources=%s remaining_sources=%s encoded_lines=%s manifest_rows=%s",
        len(completed_source_ids),
        len(remaining_scores),
        resume_state.encoded_line_count,
        resume_state.manifest_row_count,
    )
    if tokenization_processing_config.workers == 1:
        return _tokenize_missing_sources_serially(
            remaining_scores,
            paths=paths,
            output_paths=output_paths,
            state_key=state_key,
            resume_state=resume_state,
            segmentation_config=segmentation_config,
            duration_vocabulary=duration_vocabulary,
            token_vocabulary=token_vocabulary,
            difficulty_labels=difficulty_labels,
            tokenization_processing_config=tokenization_processing_config,
            show_progress=show_progress,
            profiler=profiler,
            dataset_root=dataset_root,
        )

    return _tokenize_missing_sources_in_parallel(
        remaining_scores,
        dataset_root=dataset_root,
        paths=paths,
        output_paths=output_paths,
        state_key=state_key,
        resume_state=resume_state,
        tokenization_config=tokenization_config,
        segmentation_config=segmentation_config,
        tokenization_processing_config=tokenization_processing_config,
        difficulty_labels=difficulty_labels,
        show_progress=show_progress,
        profiler=profiler,
    )


def _tokenize_missing_sources_serially(
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    *,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    output_paths: TokenizationOutputPaths,
    state_key: str,
    resume_state: TokenizationResumeState,
    segmentation_config: SegmentationConfig,
    tokenization_processing_config: TokenizationProcessingConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    difficulty_labels: dict[str, int | None] | None,
    show_progress: bool,
    profiler: ProcessingProfilerProtocol,
) -> TokenizeDatasetResult:
    encoded_line_count = resume_state.encoded_line_count
    manifest_row_count = resume_state.manifest_row_count
    encoded_count = resume_state.encoded_count
    _LOGGER.info("Running serial tokenization: sources=%s", len(parsed_scores))
    started_at = perf_counter()
    for artifact in progress(parsed_scores, description="Encoding scores", unit="score", enabled=show_progress):
        source_encoded_count, source_manifest_count, encoded_line_count = tokenize_source(
            artifact,
            dataset_root=dataset_root,
            paths=paths,
            encoded_jsonl_path=output_paths.encoded_jsonl_path,
            manifest_encoded_jsonl_path=output_paths.encoded_jsonl_path,
            encoded_manifest_path=output_paths.encoded_manifest_path,
            segmentation_config=segmentation_config,
            duration_vocabulary=duration_vocabulary,
            token_vocabulary=token_vocabulary,
            difficulty_labels=difficulty_labels,
            tokenization_processing_config=tokenization_processing_config,
            encoded_line_count=encoded_line_count,
            profiler=profiler,
        )
        encoded_count += source_encoded_count
        manifest_row_count += source_manifest_count
        append_source_completed_event(
            output_paths.state_path,
            state_key=state_key,
            source_id=artifact.source_id_value,
            encoded_line_count=encoded_line_count,
            manifest_row_count=manifest_row_count,
            encoded_count=encoded_count,
        )
        profiler.step()

    _LOGGER.info("Finished serial tokenization in %.1fs", perf_counter() - started_at)
    _LOGGER.info("Wrote encoded manifest: %s", output_paths.encoded_manifest_path)
    return _result(
        paths=paths,
        output_paths=output_paths,
        encoded_count=encoded_count,
        segment_count=manifest_row_count,
        tokenization_processing_config=tokenization_processing_config,
    )


def _tokenize_missing_sources_in_parallel(
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    *,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    output_paths: TokenizationOutputPaths,
    state_key: str,
    resume_state: TokenizationResumeState,
    tokenization_config: TokenizationConfig,
    segmentation_config: SegmentationConfig,
    tokenization_processing_config: TokenizationProcessingConfig,
    difficulty_labels: dict[str, int | None] | None,
    show_progress: bool,
    profiler: ProcessingProfilerProtocol,
) -> TokenizeDatasetResult:
    temp_root = output_paths.encoded_manifest_path.parent / "tmp"
    _LOGGER.info("Clearing tokenization temp directory: %s", temp_root)
    clear_tokenization_temp_root(temp_root)
    _LOGGER.info("Building tokenization batch tasks")
    started_at = perf_counter()
    tasks = _tokenization_batch_tasks(
        parsed_scores,
        dataset_root=dataset_root,
        paths=paths,
        output_paths=output_paths,
        temp_root=temp_root,
        tokenization_config=tokenization_config,
        segmentation_config=segmentation_config,
        tokenization_processing_config=tokenization_processing_config,
        difficulty_labels=difficulty_labels,
    )
    _LOGGER.info(
        "Built %s tokenization batch task(s) in %.1fs: sources=%s batch_size=%s workers=%s",
        len(tasks),
        perf_counter() - started_at,
        len(parsed_scores),
        tokenization_processing_config.batch_size,
        tokenization_processing_config.workers,
    )
    encoded_line_count = resume_state.encoded_line_count
    manifest_row_count = resume_state.manifest_row_count
    encoded_count = resume_state.encoded_count
    _LOGGER.info("Running tokenization batches")
    started_at = perf_counter()
    with profiler.measure("run_tokenization_batches"):
        batch_results = run_tokenization_batch_tasks(
            tasks,
            workers=tokenization_processing_config.workers,
            show_progress=show_progress,
        )
    _LOGGER.info("Finished tokenization batches in %.1fs", perf_counter() - started_at)

    _LOGGER.info("Merging %s tokenization batch result(s)", len(batch_results))
    started_at = perf_counter()
    with profiler.measure("merge_tokenization_batches"):
        for batch_result in batch_results:
            for source_result in batch_result.source_results:
                counts = merge_tokenized_source_result(
                    source_result,
                    output_paths=output_paths,
                    encoded_line_count=encoded_line_count,
                    manifest_row_count=manifest_row_count,
                    encoded_count=encoded_count,
                )
                encoded_line_count = counts.encoded_line_count
                manifest_row_count = counts.manifest_row_count
                encoded_count = counts.encoded_count
                append_source_completed_event(
                    output_paths.state_path,
                    state_key=state_key,
                    source_id=source_result.source_id_value,
                    encoded_line_count=encoded_line_count,
                    manifest_row_count=manifest_row_count,
                    encoded_count=encoded_count,
                )
                profiler.step()
    _LOGGER.info("Finished merging tokenization batches in %.1fs", perf_counter() - started_at)
    clear_tokenization_temp_root(temp_root)

    _LOGGER.info("Wrote encoded manifest: %s", output_paths.encoded_manifest_path)
    return _result(
        paths=paths,
        output_paths=output_paths,
        encoded_count=encoded_count,
        segment_count=manifest_row_count,
        tokenization_processing_config=tokenization_processing_config,
    )


def _tokenization_batch_tasks(
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    *,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    output_paths: TokenizationOutputPaths,
    temp_root: Path,
    tokenization_config: TokenizationConfig,
    segmentation_config: SegmentationConfig,
    tokenization_processing_config: TokenizationProcessingConfig,
    difficulty_labels: dict[str, int | None] | None,
) -> tuple[TokenizationBatchTask, ...]:
    batches = batched_tokenization_sources(parsed_scores, batch_size=tokenization_processing_config.batch_size)
    return tuple(
        TokenizationBatchTask(
            index=index,
            artifacts=batch,
            dataset_root=dataset_root,
            paths=paths,
            final_encoded_jsonl_path=output_paths.encoded_jsonl_path,
            temp_root=temp_root,
            segmentation_config=segmentation_config,
            tokenization_config=tokenization_config,
            tokenization_processing_config=tokenization_processing_config,
            difficulty_labels=difficulty_labels,
        )
        for index, batch in enumerate(batches)
    )


def _result(
    *,
    paths: ProcessedDatasetPaths,
    output_paths: TokenizationOutputPaths,
    encoded_count: int,
    segment_count: int,
    tokenization_processing_config: TokenizationProcessingConfig,
) -> TokenizeDatasetResult:
    return TokenizeDatasetResult(
        parsed_manifest_path=paths.parsed_manifest_path,
        encoded_manifest_path=output_paths.encoded_manifest_path,
        tokenizer_snapshot_path=output_paths.tokenizer_snapshot_path,
        encoded_count=encoded_count,
        segment_count=segment_count,
        scale_matcher_config=tokenization_processing_config.scale_matcher,
    )
