from pathlib import Path

from musak_model.data.cleaning import clean_parsed_score
from musak_model.data.config import (
    DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT,
    DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION,
    DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN,
    DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN,
    SegmentationConfig,
)
from musak_model.data.converter import PitchDegreeRegisterError
from musak_model.data.labeler import extract_difficulty_features
from musak_model.data.parser import parse_score
from musak_model.data.schema import ParsedScore, Segment, SegmentIneligibilityReason
from musak_model.data.segmenter.segmenter import segment_score
from musak_model.tokens.duration import DurationVocabulary
from musak_shared.files import collect_musicxml_files


def process_directory(
    source_directory: Path,
    duration_vocabulary: DurationVocabulary,
    *,
    segmentation_config: SegmentationConfig,
    difficulty_labels: dict[str, int | None] | None = None,
) -> list[Segment]:
    musicxml_files = collect_musicxml_files(source_directory)
    segments: list[Segment] = []
    for path in musicxml_files:
        file_segments = process_file(
            path,
            duration_vocabulary,
            segmentation_config=segmentation_config,
            difficulty_labels=difficulty_labels,
        )
        segments.extend(file_segments)

    return segments


def process_file(
    path: Path,
    duration_vocabulary: DurationVocabulary,
    *,
    segmentation_config: SegmentationConfig,
    difficulty_labels: dict[str, int | None] | None = None,
) -> list[Segment]:
    score = clean_parsed_score(parse_score(path))
    return segment_parsed_score(
        score,
        path,
        duration_vocabulary,
        segmentation_config=segmentation_config,
        difficulty_labels=difficulty_labels,
    )


def segment_parsed_score(
    score: ParsedScore,
    source_file: Path,
    duration_vocabulary: DurationVocabulary,
    *,
    segmentation_config: SegmentationConfig,
    difficulty_labels: dict[str, int | None] | None = None,
    scale_match_support_score_margin: float = DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN,
    scale_match_selection_score_margin: float = DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN,
    scale_match_maximum_unexplained_weight_fraction: float = DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION,
    scale_match_maximum_explanation_pitch_class_count: int = DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT,
) -> list[Segment]:
    difficulty_level = _resolve_difficulty_level(
        source_file,
        difficulty_labels=difficulty_labels,
    )

    segments = segment_score(
        score,
        source_file,
        duration_vocabulary=duration_vocabulary,
        segmentation=segmentation_config,
        difficulty_level=difficulty_level,
        scale_match_support_score_margin=scale_match_support_score_margin,
        scale_match_selection_score_margin=scale_match_selection_score_margin,
        scale_match_maximum_unexplained_weight_fraction=scale_match_maximum_unexplained_weight_fraction,
        scale_match_maximum_explanation_pitch_class_count=scale_match_maximum_explanation_pitch_class_count,
    )

    return [
        _attach_difficulty_features(
            segment,
            score=score,
            duration_vocabulary=duration_vocabulary,
        )
        for segment in segments
    ]


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
