from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from musak_model.data.config import SegmentationConfig
from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.processing.config import ProcessingConfig, TokenizationProcessingConfig
from musak_model.processing.manifest import ParsedManifestField, ParsedManifestStatus, read_parsed_manifest
from musak_model.processing.parser import (
    ParseDatasetResult,
    ParsedScoreArtifact,
    load_parsed_score_artifacts,
    parse_dataset,
)
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.processing.tokenizer import TokenizeDatasetResult, tokenize_parsed_scores
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_shared.profiling import NULL_PROFILER, ProfilerProtocol

type ProcessingStage = Literal["parse", "tokenize", "process"]
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessDatasetResult:
    parsed_manifest_path: Path
    encoded_manifest_path: Path | None
    tokenizer_snapshot_path: Path | None
    parsed_count: int
    encoded_count: int
    error_count: int
    scale_matcher_config: ScaleMatcherConfig


def process_dataset(
    dataset_root: Path,
    *,
    processed_root: Path,
    segmentation_config: SegmentationConfig,
    tokenization_config: TokenizationConfig,
    processing_config: ProcessingConfig,
    stage: ProcessingStage,
    difficulty_labels: dict[str, int | None] | None = None,
    overwrite: bool = False,
    show_progress: bool = False,
    profiler: ProfilerProtocol = NULL_PROFILER,
) -> ProcessDatasetResult:
    match stage:
        case "parse":
            _LOGGER.info("Starting parse stage")
            started_at = perf_counter()
            parse_result = parse_dataset(
                dataset_root,
                processed_root=processed_root,
                overwrite=overwrite,
                workers=processing_config.parsing.workers,
                show_progress=show_progress,
                profiler=profiler,
            )
            _LOGGER.info("Finished parse stage in %.1fs", perf_counter() - started_at)
            return _parse_result_to_process_result(
                parse_result,
                tokenization_processing_config=processing_config.tokenization,
            )
        case "tokenize":
            _LOGGER.info("Loading parsed artifacts for tokenization")
            started_at = perf_counter()
            parsed_scores = load_parsed_score_artifacts(dataset_root, processed_root=processed_root)
            parse_counts = _parse_counts(dataset_root, processed_root=processed_root)
            _LOGGER.info("Loaded %s parsed artifact(s) in %.1fs", len(parsed_scores), perf_counter() - started_at)
            _LOGGER.info("Starting tokenize stage")
            started_at = perf_counter()
            tokenize_result = _tokenize_existing_parsed_scores(
                parsed_scores,
                dataset_root=dataset_root,
                processed_root=processed_root,
                segmentation_config=segmentation_config,
                tokenization_config=tokenization_config,
                tokenization_processing_config=processing_config.tokenization,
                difficulty_labels=difficulty_labels,
                overwrite=overwrite,
                show_progress=show_progress,
                profiler=profiler,
            )
            _LOGGER.info("Finished tokenize stage in %.1fs", perf_counter() - started_at)
            return _tokenize_result_to_process_result(
                tokenize_result,
                parsed_count=parse_counts.parsed_count,
                error_count=parse_counts.error_count,
            )
        case "process":
            _LOGGER.info("Starting parse stage")
            started_at = perf_counter()
            parse_result = parse_dataset(
                dataset_root,
                processed_root=processed_root,
                overwrite=overwrite,
                workers=processing_config.parsing.workers,
                show_progress=show_progress,
                profiler=profiler,
            )
            _LOGGER.info("Finished parse stage in %.1fs", perf_counter() - started_at)
            _LOGGER.info("Loading parsed artifacts for tokenization")
            started_at = perf_counter()
            parsed_scores = load_parsed_score_artifacts(dataset_root, processed_root=processed_root)
            _LOGGER.info("Loaded %s parsed artifact(s) in %.1fs", len(parsed_scores), perf_counter() - started_at)
            _LOGGER.info("Starting tokenize stage")
            started_at = perf_counter()
            tokenize_result = _tokenize_existing_parsed_scores(
                parsed_scores,
                dataset_root=dataset_root,
                processed_root=processed_root,
                segmentation_config=segmentation_config,
                tokenization_config=tokenization_config,
                tokenization_processing_config=processing_config.tokenization,
                difficulty_labels=difficulty_labels,
                overwrite=overwrite,
                show_progress=show_progress,
                profiler=profiler,
            )
            _LOGGER.info("Finished tokenize stage in %.1fs", perf_counter() - started_at)
            return _tokenize_result_to_process_result(
                tokenize_result,
                parsed_count=parse_result.parsed_count,
                error_count=parse_result.error_count,
            )


@dataclass(frozen=True)
class _ParseCounts:
    parsed_count: int
    error_count: int


def _tokenize_existing_parsed_scores(
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    *,
    dataset_root: Path,
    processed_root: Path,
    segmentation_config: SegmentationConfig,
    tokenization_config: TokenizationConfig,
    tokenization_processing_config: TokenizationProcessingConfig,
    difficulty_labels: dict[str, int | None] | None,
    overwrite: bool,
    show_progress: bool,
    profiler: ProfilerProtocol,
) -> TokenizeDatasetResult:
    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
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


def _parse_counts(dataset_root: Path, *, processed_root: Path) -> _ParseCounts:
    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    rows = read_parsed_manifest(paths.parsed_manifest_path)
    parsed_count = sum(row[ParsedManifestField.STATUS] == ParsedManifestStatus.SUCCESS.value for row in rows)
    error_count = sum(row[ParsedManifestField.STATUS] == ParsedManifestStatus.ERROR.value for row in rows)
    return _ParseCounts(parsed_count=parsed_count, error_count=error_count)


def _parse_result_to_process_result(
    parse_result: ParseDatasetResult,
    *,
    tokenization_processing_config: TokenizationProcessingConfig,
) -> ProcessDatasetResult:
    return ProcessDatasetResult(
        parsed_manifest_path=parse_result.parsed_manifest_path,
        encoded_manifest_path=None,
        tokenizer_snapshot_path=None,
        parsed_count=parse_result.parsed_count,
        encoded_count=0,
        error_count=parse_result.error_count,
        scale_matcher_config=tokenization_processing_config.scale_matcher,
    )


def _tokenize_result_to_process_result(
    tokenize_result: TokenizeDatasetResult,
    *,
    parsed_count: int,
    error_count: int,
) -> ProcessDatasetResult:
    return ProcessDatasetResult(
        parsed_manifest_path=tokenize_result.parsed_manifest_path,
        encoded_manifest_path=tokenize_result.encoded_manifest_path,
        tokenizer_snapshot_path=tokenize_result.tokenizer_snapshot_path,
        parsed_count=parsed_count,
        encoded_count=tokenize_result.encoded_count,
        error_count=error_count,
        scale_matcher_config=tokenize_result.scale_matcher_config,
    )
