from pathlib import Path

from musak_model.data.cleaning import clean_parsed_score
from musak_model.data.config import SegmentationConfig
from musak_model.data.converter import PitchDegreeRegisterError
from musak_model.data.labeler import extract_difficulty_features
from musak_model.data.parser import parse_score
from musak_model.data.schema import ParsedScore, Segment, SegmentIneligibilityReason
from musak_model.data.segmenter import segment_score
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_shared.files import collect_musicxml_files


def process_directory(
    source_directory: Path,
    *,
    segmentation: SegmentationConfig,
    difficulty_labels: dict[str, int] | None = None,
) -> list[Segment]:
    duration_vocabulary = DurationVocabulary(TokenizationConfig.load())
    musicxml_files = collect_musicxml_files(source_directory)
    segments: list[Segment] = []
    for path in musicxml_files:
        file_segments = process_file(
            path,
            segmentation=segmentation,
            difficulty_labels=difficulty_labels,
            duration_vocabulary=duration_vocabulary,
        )
        segments.extend(file_segments)

    return segments


def process_file(
    path: Path,
    *,
    segmentation: SegmentationConfig,
    difficulty_labels: dict[str, int] | None = None,
    duration_vocabulary: DurationVocabulary | None = None,
) -> list[Segment]:
    score = clean_parsed_score(parse_score(path))
    return segment_parsed_score(
        score,
        path,
        segmentation=segmentation,
        difficulty_labels=difficulty_labels,
        duration_vocabulary=duration_vocabulary,
    )


def segment_parsed_score(
    score: ParsedScore,
    source_file: Path,
    *,
    segmentation: SegmentationConfig,
    difficulty_labels: dict[str, int] | None = None,
    duration_vocabulary: DurationVocabulary | None = None,
) -> list[Segment]:
    resolved_duration_vocabulary = duration_vocabulary or DurationVocabulary(TokenizationConfig.load())
    difficulty_level = _resolve_difficulty_level(source_file, difficulty_labels=difficulty_labels)

    segments = segment_score(
        score,
        source_file,
        scale_type=score.scale_type,
        duration_vocabulary=resolved_duration_vocabulary,
        segmentation=segmentation,
        difficulty_level=difficulty_level,
    )

    return [
        _attach_difficulty_features(
            segment,
            score=score,
            duration_vocabulary=resolved_duration_vocabulary,
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
    difficulty_labels: dict[str, int] | None,
) -> int | None:
    if difficulty_labels is None:
        return None

    return difficulty_labels.get(path.stem)
