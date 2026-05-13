from pathlib import Path

from musak_model.common.files import collect_musicxml_files
from musak_model.data.labeler import extract_difficulty_features
from musak_model.data.parser import parse_score
from musak_model.data.schema import ParsedScore, Segment
from musak_model.data.segmenter import segment_score
from musak_model.tokens.schema import ScaleType

_MODE_TO_SCALE_TYPE: dict[str, ScaleType] = {
    "major": ScaleType.MAJOR,
    "minor": ScaleType.NATURAL_MINOR,
}


def process_directory(
    source_dir: Path,
    *,
    window_bars: int,
    stride_bars: int,
    difficulty_labels: dict[str, int] | None = None,
) -> list[Segment]:
    musicxml_files = collect_musicxml_files(source_dir)
    segments: list[Segment] = []
    for path in musicxml_files:
        file_segments = process_file(
            path,
            window_bars=window_bars,
            stride_bars=stride_bars,
            difficulty_labels=difficulty_labels,
        )
        segments.extend(file_segments)

    return segments


def process_file(
    path: Path,
    *,
    window_bars: int,
    stride_bars: int,
    difficulty_labels: dict[str, int] | None = None,
) -> list[Segment]:
    score = parse_score(path)
    scale_type = _resolve_scale_type(score.mode)
    difficulty_level = _resolve_difficulty_level(path, difficulty_labels=difficulty_labels)

    segments = segment_score(
        score,
        path,
        scale_type=scale_type,
        window_bars=window_bars,
        stride_bars=stride_bars,
        difficulty_level=difficulty_level,
    )

    return [
        _attach_difficulty_features(
            segment,
            score=score,
        )
        for segment in segments
    ]


def _attach_difficulty_features(
    segment: Segment,
    *,
    score: ParsedScore,
) -> Segment:
    features = extract_difficulty_features(segment, score=score, scale_type=segment.scale_type)
    metadata = segment.metadata.model_copy(update={"difficulty_features": features})
    return segment.model_copy(update={"metadata": metadata})


def _resolve_scale_type(mode: str) -> ScaleType:
    scale_type = _MODE_TO_SCALE_TYPE.get(mode)
    if scale_type is None:
        raise ValueError(f"unsupported mode '{mode}'; extend _MODE_TO_SCALE_TYPE or pass scale_type explicitly")

    return scale_type


def _resolve_difficulty_level(
    path: Path,
    *,
    difficulty_labels: dict[str, int] | None,
) -> int | None:
    if difficulty_labels is None:
        return None

    return difficulty_labels.get(path.stem)
