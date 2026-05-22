from collections.abc import Iterator
from pathlib import Path

from musak_model.data.cleaning import clean_parsed_score
from musak_model.data.config import SegmentationConfig
from musak_model.data.converter import PitchDegreeRegisterError
from musak_model.data.labeler import extract_difficulty_features
from musak_model.data.parser import parse_score
from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.data.schema import ParsedScore, Segment, SegmentIneligibilityReason
from musak_model.data.segmenter.segmenter import iter_score_segments
from musak_model.processing.profiler import NULL_PROCESSING_PROFILER, ProcessingProfilerProtocol
from musak_model.tokens.duration import DurationVocabulary
from musak_shared.files import collect_musicxml_files


def process_directory(
    source_directory: Path,
    duration_vocabulary: DurationVocabulary,
    *,
    segmentation_config: SegmentationConfig,
    scale_matcher_config: ScaleMatcherConfig,
    difficulty_labels: dict[str, int | None] | None = None,
    profiler: ProcessingProfilerProtocol = NULL_PROCESSING_PROFILER,
) -> list[Segment]:
    musicxml_files = collect_musicxml_files(source_directory)
    segments: list[Segment] = []
    for path in musicxml_files:
        file_segments = process_file(
            path,
            duration_vocabulary,
            segmentation_config=segmentation_config,
            scale_matcher_config=scale_matcher_config,
            difficulty_labels=difficulty_labels,
            profiler=profiler,
        )
        segments.extend(file_segments)

    return segments


def process_file(
    path: Path,
    duration_vocabulary: DurationVocabulary,
    *,
    segmentation_config: SegmentationConfig,
    scale_matcher_config: ScaleMatcherConfig,
    difficulty_labels: dict[str, int | None] | None = None,
    profiler: ProcessingProfilerProtocol = NULL_PROCESSING_PROFILER,
) -> list[Segment]:
    score = clean_parsed_score(parse_score(path))
    return segment_parsed_score(
        score,
        path,
        duration_vocabulary,
        segmentation_config=segmentation_config,
        scale_matcher_config=scale_matcher_config,
        difficulty_labels=difficulty_labels,
        profiler=profiler,
    )


def segment_parsed_score(
    score: ParsedScore,
    source_file: Path,
    duration_vocabulary: DurationVocabulary,
    *,
    segmentation_config: SegmentationConfig,
    scale_matcher_config: ScaleMatcherConfig,
    difficulty_labels: dict[str, int | None] | None = None,
    profiler: ProcessingProfilerProtocol = NULL_PROCESSING_PROFILER,
) -> list[Segment]:
    return list(
        iter_segment_parsed_score(
            score,
            source_file,
            duration_vocabulary,
            segmentation_config=segmentation_config,
            scale_matcher_config=scale_matcher_config,
            difficulty_labels=difficulty_labels,
            profiler=profiler,
        )
    )


def iter_segment_parsed_score(
    score: ParsedScore,
    source_file: Path,
    duration_vocabulary: DurationVocabulary,
    *,
    segmentation_config: SegmentationConfig,
    scale_matcher_config: ScaleMatcherConfig,
    difficulty_labels: dict[str, int | None] | None = None,
    profiler: ProcessingProfilerProtocol = NULL_PROCESSING_PROFILER,
) -> Iterator[Segment]:
    difficulty_level = _resolve_difficulty_level(
        source_file,
        difficulty_labels=difficulty_labels,
    )

    segments = iter_score_segments(
        score,
        source_file,
        duration_vocabulary=duration_vocabulary,
        segmentation=segmentation_config,
        difficulty_level=difficulty_level,
        scale_matcher_config=scale_matcher_config,
        profiler=profiler,
    )

    yield from _attach_difficulty_features_to_segments(
        segments,
        score=score,
        source_file=source_file,
        duration_vocabulary=duration_vocabulary,
        profiler=profiler,
    )


def _attach_difficulty_features_to_segments(
    segments: Iterator[Segment],
    *,
    score: ParsedScore,
    source_file: Path,
    duration_vocabulary: DurationVocabulary,
    profiler: ProcessingProfilerProtocol,
) -> Iterator[Segment]:
    for segment in segments:
        yield _attach_difficulty_features_with_profile(
            segment,
            score=score,
            source_file=source_file,
            duration_vocabulary=duration_vocabulary,
            profiler=profiler,
        )


def _attach_difficulty_features_with_profile(
    segment: Segment,
    *,
    score: ParsedScore,
    source_file: Path,
    duration_vocabulary: DurationVocabulary,
    profiler: ProcessingProfilerProtocol,
) -> Segment:
    with profiler.measure("difficulty_features", source_file=source_file):
        return _attach_difficulty_features(
            segment,
            score=score,
            duration_vocabulary=duration_vocabulary,
        )


def _attach_difficulty_features(
    segment: Segment,
    *,
    score: ParsedScore,
    duration_vocabulary: DurationVocabulary,
) -> Segment:
    if not segment.metadata.eligible_for_training:
        return segment

    try:
        features = extract_difficulty_features(
            segment,
            score=score,
            scale_type=segment.scale_type,
            duration_vocabulary=duration_vocabulary,
        )
    except PitchDegreeRegisterError:
        metadata = segment.metadata.model_copy(
            update={
                "eligible_for_training": False,
                "ineligibility_reasons": segment.metadata.ineligibility_reasons
                | {SegmentIneligibilityReason.REGISTER_OUT_OF_RANGE},
            }
        )
        return segment.model_copy(update={"metadata": metadata})

    metadata = segment.metadata.model_copy(update={"difficulty_features": features})
    return segment.model_copy(update={"metadata": metadata})


def _resolve_difficulty_level(
    path: Path,
    *,
    difficulty_labels: dict[str, int | None] | None,
) -> int | None:
    if difficulty_labels is None:
        return None

    for key in _difficulty_label_keys(path):
        if key in difficulty_labels:
            return difficulty_labels[key]

    return None


def _difficulty_label_keys(path: Path) -> tuple[str, ...]:
    return (path.as_posix(), path.name, path.stem)
