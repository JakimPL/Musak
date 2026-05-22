from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from musak_model.data.config import (
    DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT,
    DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION,
    DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN,
    DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN,
    DataProcessingConfig,
    SegmentationConfig,
)
from musak_model.processing.manifest import ParsedManifestField, ParsedManifestStatus, read_parsed_manifest
from musak_model.processing.parser import (
    ParseDatasetResult,
    ParsedScoreArtifact,
    load_parsed_score_artifacts,
    parse_dataset,
)
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.profiler import NULL_PROCESSING_PROFILER, ProcessingProfilerProtocol
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.processing.tokenizer import TokenizeDatasetResult, tokenize_parsed_scores
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary

type ProcessingStage = Literal["parse", "tokenize", "process"]


@dataclass(frozen=True)
class ProcessDatasetResult:
    parsed_manifest_path: Path
    encoded_manifest_path: Path | None
    tokenizer_snapshot_path: Path | None
    parsed_count: int
    encoded_count: int
    error_count: int
    scale_match_support_score_margin: float = DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN
    scale_match_selection_score_margin: float = DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN
    scale_match_maximum_unexplained_weight_fraction: float = DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION
    scale_match_maximum_explanation_pitch_class_count: int = DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT


def process_dataset(
    dataset_root: Path,
    *,
    processed_root: Path,
    segmentation_config: SegmentationConfig,
    tokenization_config: TokenizationConfig,
    data_processing_config: DataProcessingConfig,
    stage: ProcessingStage,
    difficulty_labels: dict[str, int | None] | None = None,
    overwrite: bool = False,
    workers: int = 1,
    show_progress: bool = False,
    profiler: ProcessingProfilerProtocol = NULL_PROCESSING_PROFILER,
) -> ProcessDatasetResult:
    match stage:
        case "parse":
            parse_result = parse_dataset(
                dataset_root,
                processed_root=processed_root,
                overwrite=overwrite,
                workers=workers,
                show_progress=show_progress,
                profiler=profiler,
            )
            return _parse_result_to_process_result(
                parse_result,
                data_processing_config=data_processing_config,
            )
        case "tokenize":
            parsed_scores = load_parsed_score_artifacts(dataset_root, processed_root=processed_root)
            parse_counts = _parse_counts(dataset_root, processed_root=processed_root)
            tokenize_result = _tokenize_existing_parsed_scores(
                parsed_scores,
                dataset_root=dataset_root,
                processed_root=processed_root,
                segmentation_config=segmentation_config,
                tokenization_config=tokenization_config,
                data_processing_config=data_processing_config,
                difficulty_labels=difficulty_labels,
                overwrite=overwrite,
                show_progress=show_progress,
                profiler=profiler,
            )
            return _tokenize_result_to_process_result(
                tokenize_result,
                parsed_count=parse_counts.parsed_count,
                error_count=parse_counts.error_count,
            )
        case "process":
            parse_result = parse_dataset(
                dataset_root,
                processed_root=processed_root,
                overwrite=overwrite,
                workers=workers,
                show_progress=show_progress,
                profiler=profiler,
            )
            tokenize_result = _tokenize_existing_parsed_scores(
                parse_result.parsed_scores,
                dataset_root=dataset_root,
                processed_root=processed_root,
                segmentation_config=segmentation_config,
                tokenization_config=tokenization_config,
                data_processing_config=data_processing_config,
                difficulty_labels=difficulty_labels,
                overwrite=overwrite,
                show_progress=show_progress,
                profiler=profiler,
            )
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
    data_processing_config: DataProcessingConfig,
    difficulty_labels: dict[str, int | None] | None,
    overwrite: bool,
    show_progress: bool,
    profiler: ProcessingProfilerProtocol,
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
        segmentation_config=segmentation_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        difficulty_labels=difficulty_labels,
        data_processing_config=data_processing_config,
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
    data_processing_config: DataProcessingConfig,
) -> ProcessDatasetResult:
    return ProcessDatasetResult(
        parsed_manifest_path=parse_result.parsed_manifest_path,
        encoded_manifest_path=None,
        tokenizer_snapshot_path=None,
        parsed_count=parse_result.parsed_count,
        encoded_count=0,
        error_count=parse_result.error_count,
        scale_match_support_score_margin=data_processing_config.scale_match_support_score_margin,
        scale_match_selection_score_margin=data_processing_config.scale_match_selection_score_margin,
        scale_match_maximum_unexplained_weight_fraction=(
            data_processing_config.scale_match_maximum_unexplained_weight_fraction
        ),
        scale_match_maximum_explanation_pitch_class_count=(
            data_processing_config.scale_match_maximum_explanation_pitch_class_count
        ),
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
        scale_match_support_score_margin=tokenize_result.scale_match_support_score_margin,
        scale_match_selection_score_margin=tokenize_result.scale_match_selection_score_margin,
        scale_match_maximum_unexplained_weight_fraction=(
            tokenize_result.scale_match_maximum_unexplained_weight_fraction
        ),
        scale_match_maximum_explanation_pitch_class_count=(
            tokenize_result.scale_match_maximum_explanation_pitch_class_count
        ),
    )
