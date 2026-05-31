from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from musak_model.data.config import SegmentationConfig
from musak_model.data.pipeline import iter_segment_parsed_score
from musak_model.data.schema import Segment, SegmentIneligibilityReason
from musak_model.evaluation.diagnostics import SegmentDiagnostics, diagnose_segment
from musak_model.processing.config import TokenizationProcessingConfig
from musak_model.processing.io import load_parsed_score_json
from musak_model.processing.manifest import encoded_row
from musak_model.processing.parser import ParsedScoreArtifact
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.tokenizer.output import EncodedManifestAppender, append_jsonl_model
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.split import _encode_segment
from musak_shared.profiling import ProfilerProtocol

if TYPE_CHECKING:
    from musak_model.training.ingestion.schema import EncodedExercise


@dataclass(frozen=True)
class ProcessedSegment:
    segment: Segment
    diagnostics: SegmentDiagnostics
    encoded_sample: "EncodedExercise | None"
    encoded_line: int | None
    next_encoded_line_count: int


def tokenize_source(
    artifact: ParsedScoreArtifact,
    *,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    encoded_jsonl_path: Path,
    manifest_encoded_jsonl_path: Path,
    encoded_manifest_path: Path,
    segmentation_config: SegmentationConfig,
    tokenization_processing_config: TokenizationProcessingConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    difficulty_labels: dict[str, int | None] | None,
    encoded_line_count: int,
    profiler: ProfilerProtocol,
) -> tuple[int, int, int]:
    source_metadata_path = Path(artifact.source_path.resolve().relative_to(dataset_root.resolve()).as_posix())
    segments = _segment_artifact(
        artifact,
        source_metadata_path=source_metadata_path,
        duration_vocabulary=duration_vocabulary,
        segmentation_config=segmentation_config,
        difficulty_labels=difficulty_labels,
        tokenization_processing_config=tokenization_processing_config,
        profiler=profiler,
    )
    return _write_source_outputs(
        segments,
        artifact=artifact,
        source_metadata_path=source_metadata_path,
        dataset_root=dataset_root,
        paths=paths,
        encoded_jsonl_path=encoded_jsonl_path,
        manifest_encoded_jsonl_path=manifest_encoded_jsonl_path,
        encoded_manifest_path=encoded_manifest_path,
        segmentation_config=segmentation_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        tokenization_processing_config=tokenization_processing_config,
        encoded_line_count=encoded_line_count,
        profiler=profiler,
    )


def _segment_artifact(
    artifact: ParsedScoreArtifact,
    *,
    source_metadata_path: Path,
    duration_vocabulary: DurationVocabulary,
    segmentation_config: SegmentationConfig,
    tokenization_processing_config: TokenizationProcessingConfig,
    difficulty_labels: dict[str, int | None] | None,
    profiler: ProfilerProtocol,
) -> Iterator[Segment]:
    score = load_parsed_score_json(artifact.parsed_path)
    return iter_segment_parsed_score(
        score,
        source_metadata_path,
        duration_vocabulary,
        segmentation_config=segmentation_config,
        difficulty_labels=difficulty_labels,
        scale_matcher_config=tokenization_processing_config.scale_matcher,
        profiler=profiler,
    )


def _write_source_outputs(
    segments: Iterator[Segment],
    *,
    artifact: ParsedScoreArtifact,
    source_metadata_path: Path,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    encoded_jsonl_path: Path,
    manifest_encoded_jsonl_path: Path,
    encoded_manifest_path: Path,
    segmentation_config: SegmentationConfig,
    tokenization_processing_config: TokenizationProcessingConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    encoded_line_count: int,
    profiler: ProfilerProtocol,
) -> tuple[int, int, int]:
    source_encoded_count = 0
    source_manifest_count = 0
    with EncodedManifestAppender(encoded_manifest_path) as manifest_appender:
        for segment in segments:
            processed_segment = _process_segment(
                segment,
                source_metadata_path=source_metadata_path,
                encoded_jsonl_path=encoded_jsonl_path,
                duration_vocabulary=duration_vocabulary,
                token_vocabulary=token_vocabulary,
                tokenization_processing_config=tokenization_processing_config,
                encoded_line_count=encoded_line_count,
                profiler=profiler,
            )
            encoded_line_count = processed_segment.next_encoded_line_count
            if processed_segment.encoded_line is not None:
                source_encoded_count += 1

            row = _manifest_row(
                artifact=artifact,
                segment=processed_segment.segment,
                diagnostics=processed_segment.diagnostics,
                dataset_root=dataset_root,
                paths=paths,
                encoded_jsonl_path=manifest_encoded_jsonl_path,
                encoded_sample=processed_segment.encoded_sample,
                encoded_line=processed_segment.encoded_line,
                segmentation_config=segmentation_config,
                profiler=profiler,
                source_metadata_path=source_metadata_path,
            )
            with profiler.measure("append_encoded_manifest", source_file=source_metadata_path):
                manifest_appender.append(row)
            source_manifest_count += 1

    return source_encoded_count, source_manifest_count, encoded_line_count


def _process_segment(
    segment: Segment,
    *,
    source_metadata_path: Path,
    encoded_jsonl_path: Path,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    tokenization_processing_config: TokenizationProcessingConfig,
    encoded_line_count: int,
    profiler: ProfilerProtocol,
) -> ProcessedSegment:
    with profiler.measure("diagnose_segment", source_file=source_metadata_path):
        diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)

    with profiler.measure("apply_processing_filters", source_file=source_metadata_path):
        segment = _apply_processing_filters(
            segment,
            diagnostics=diagnostics,
            tokenization_processing_config=tokenization_processing_config,
        )
    if not segment.metadata.eligible_for_training:
        return ProcessedSegment(
            segment=segment,
            diagnostics=diagnostics,
            encoded_sample=None,
            encoded_line=None,
            next_encoded_line_count=encoded_line_count,
        )

    with profiler.measure("encode_segment", source_file=source_metadata_path):
        encoded_sample = _encode_segment(segment, token_vocabulary=token_vocabulary)

    with profiler.measure("append_encoded_jsonl", source_file=source_metadata_path):
        encoded_line = append_jsonl_model(encoded_sample, encoded_jsonl_path, line_index=encoded_line_count)

    return ProcessedSegment(
        segment=segment,
        diagnostics=diagnostics,
        encoded_sample=encoded_sample,
        encoded_line=encoded_line,
        next_encoded_line_count=encoded_line_count + 1,
    )


def _manifest_row(
    *,
    artifact: ParsedScoreArtifact,
    segment: Segment,
    diagnostics: SegmentDiagnostics,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    encoded_jsonl_path: Path,
    encoded_sample: "EncodedExercise | None",
    encoded_line: int | None,
    segmentation_config: SegmentationConfig,
    profiler: ProfilerProtocol,
    source_metadata_path: Path,
) -> dict[str, object]:
    with profiler.measure("encoded_manifest_row", source_file=source_metadata_path):
        return encoded_row(
            source_id_value=artifact.source_id_value,
            source_path=artifact.source_path,
            dataset_root=dataset_root,
            parsed_path=artifact.parsed_path,
            processed_root=paths.root,
            segment=segment,
            diagnostics=diagnostics,
            encoded_sample=encoded_sample,
            encoded_shard=encoded_jsonl_path,
            encoded_line=encoded_line,
            segmentation_mode=segmentation_config.mode,
        )


def _apply_processing_filters(
    segment: Segment,
    *,
    diagnostics: SegmentDiagnostics,
    tokenization_processing_config: TokenizationProcessingConfig,
) -> Segment:
    if not tokenization_processing_config.remove_segments_with_silent_bars:
        return segment

    if diagnostics.silent_bar_count == 0:
        return segment

    metadata = segment.metadata.model_copy(
        update={
            "eligible_for_training": False,
            "ineligibility_reasons": segment.metadata.ineligibility_reasons
            | frozenset({SegmentIneligibilityReason.SILENT_BAR}),
        }
    )

    return segment.model_copy(update={"metadata": metadata})
