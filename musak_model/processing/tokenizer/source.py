from pathlib import Path
from typing import TYPE_CHECKING

from musak_model.data.config import DataProcessingConfig, SegmentationConfig
from musak_model.data.pipeline import segment_parsed_score
from musak_model.data.schema import Segment, SegmentIneligibilityReason
from musak_model.evaluation.diagnostics import diagnose_segment
from musak_model.processing.manifest import encoded_row
from musak_model.processing.parser import ParsedScoreArtifact
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.profiler import ProcessingProfilerProtocol
from musak_model.processing.tokenizer.output import append_encoded_manifest_rows, append_jsonl_model
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.split import _encode_segment

if TYPE_CHECKING:
    from musak_model.training.ingestion.schema import EncodedExercise


def tokenize_source(
    artifact: ParsedScoreArtifact,
    *,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    encoded_jsonl_path: Path,
    encoded_manifest_path: Path,
    segmentation_config: SegmentationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    difficulty_labels: dict[str, int | None] | None,
    data_processing_config: DataProcessingConfig,
    encoded_line_count: int,
    profiler: ProcessingProfilerProtocol,
) -> tuple[int, int, int]:
    source_metadata_path = Path(artifact.source_path.resolve().relative_to(dataset_root.resolve()).as_posix())
    segments = _segment_artifact(
        artifact,
        source_metadata_path=source_metadata_path,
        duration_vocabulary=duration_vocabulary,
        segmentation_config=segmentation_config,
        difficulty_labels=difficulty_labels,
        data_processing_config=data_processing_config,
        profiler=profiler,
    )
    return _write_source_outputs(
        segments,
        artifact=artifact,
        source_metadata_path=source_metadata_path,
        dataset_root=dataset_root,
        paths=paths,
        encoded_jsonl_path=encoded_jsonl_path,
        encoded_manifest_path=encoded_manifest_path,
        segmentation_config=segmentation_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        data_processing_config=data_processing_config,
        encoded_line_count=encoded_line_count,
        profiler=profiler,
    )


def _segment_artifact(
    artifact: ParsedScoreArtifact,
    *,
    source_metadata_path: Path,
    duration_vocabulary: DurationVocabulary,
    segmentation_config: SegmentationConfig,
    difficulty_labels: dict[str, int | None] | None,
    data_processing_config: DataProcessingConfig,
    profiler: ProcessingProfilerProtocol,
) -> list[Segment]:
    return segment_parsed_score(
        artifact.score,
        source_metadata_path,
        duration_vocabulary,
        segmentation_config=segmentation_config,
        difficulty_labels=difficulty_labels,
        scale_match_support_score_margin=data_processing_config.scale_match_support_score_margin,
        scale_match_selection_score_margin=data_processing_config.scale_match_selection_score_margin,
        scale_match_maximum_unexplained_weight_fraction=(
            data_processing_config.scale_match_maximum_unexplained_weight_fraction
        ),
        scale_match_maximum_explanation_pitch_class_count=(
            data_processing_config.scale_match_maximum_explanation_pitch_class_count
        ),
        profiler=profiler,
    )


def _write_source_outputs(
    segments: list[Segment],
    *,
    artifact: ParsedScoreArtifact,
    source_metadata_path: Path,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    encoded_jsonl_path: Path,
    encoded_manifest_path: Path,
    segmentation_config: SegmentationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    data_processing_config: DataProcessingConfig,
    encoded_line_count: int,
    profiler: ProcessingProfilerProtocol,
) -> tuple[int, int, int]:
    source_rows: list[dict[str, object]] = []
    source_encoded_count = 0
    for segment in segments:
        segment, encoded_sample, encoded_line, encoded_line_count = _encode_segment_if_eligible(
            segment,
            source_metadata_path=source_metadata_path,
            encoded_jsonl_path=encoded_jsonl_path,
            duration_vocabulary=duration_vocabulary,
            token_vocabulary=token_vocabulary,
            data_processing_config=data_processing_config,
            encoded_line_count=encoded_line_count,
            profiler=profiler,
        )
        if encoded_line is not None:
            source_encoded_count += 1

        source_rows.append(
            _manifest_row(
                artifact=artifact,
                segment=segment,
                dataset_root=dataset_root,
                paths=paths,
                encoded_jsonl_path=encoded_jsonl_path,
                encoded_sample=encoded_sample,
                encoded_line=encoded_line,
                segmentation_config=segmentation_config,
                duration_vocabulary=duration_vocabulary,
                profiler=profiler,
                source_metadata_path=source_metadata_path,
            )
        )

    with profiler.measure("append_encoded_manifest", source_file=source_metadata_path):
        append_encoded_manifest_rows(source_rows, encoded_manifest_path)

    return source_encoded_count, len(source_rows), encoded_line_count


def _encode_segment_if_eligible(
    segment: Segment,
    *,
    source_metadata_path: Path,
    encoded_jsonl_path: Path,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    data_processing_config: DataProcessingConfig,
    encoded_line_count: int,
    profiler: ProcessingProfilerProtocol,
) -> tuple[Segment, "EncodedExercise | None", int | None, int]:
    with profiler.measure("apply_processing_filters", source_file=source_metadata_path):
        segment = _apply_processing_filters(
            segment,
            duration_vocabulary=duration_vocabulary,
            data_processing_config=data_processing_config,
        )
    if not segment.metadata.eligible_for_training:
        return segment, None, None, encoded_line_count

    with profiler.measure("encode_segment", source_file=source_metadata_path):
        encoded_sample = _encode_segment(segment, token_vocabulary=token_vocabulary)

    with profiler.measure("append_encoded_jsonl", source_file=source_metadata_path):
        encoded_line = append_jsonl_model(encoded_sample, encoded_jsonl_path, line_index=encoded_line_count)

    return segment, encoded_sample, encoded_line, encoded_line_count + 1


def _manifest_row(
    *,
    artifact: ParsedScoreArtifact,
    segment: Segment,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
    encoded_jsonl_path: Path,
    encoded_sample: "EncodedExercise | None",
    encoded_line: int | None,
    segmentation_config: SegmentationConfig,
    duration_vocabulary: DurationVocabulary,
    profiler: ProcessingProfilerProtocol,
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
            duration_vocabulary=duration_vocabulary,
            encoded_sample=encoded_sample,
            encoded_shard=encoded_jsonl_path,
            encoded_line=encoded_line,
            segmentation_mode=segmentation_config.mode,
        )


def _apply_processing_filters(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    data_processing_config: DataProcessingConfig,
) -> Segment:
    if not data_processing_config.remove_segments_with_silent_bars:
        return segment

    diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)
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
