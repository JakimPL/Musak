from pathlib import Path

from musak_model.data.config import SegmentationConfig
from musak_model.data.schema import ParsedScore, Segment
from musak_model.data.segmenter.streams import tokenize_unified_stream_safely
from musak_model.data.segmenter.windows import create_windows
from musak_model.tokens.duration import DurationVocabulary


def segment_score(
    score: ParsedScore,
    source_file: Path,
    *,
    duration_vocabulary: DurationVocabulary,
    segmentation: SegmentationConfig,
    difficulty_level: int | None = None,
) -> list[Segment]:
    unified_tokens = tokenize_unified_stream_safely(
        score=score,
        duration_vocabulary=duration_vocabulary,
    )

    return create_windows(
        unified_tokens=unified_tokens,
        score=score,
        source_file=source_file,
        segmentation=segmentation,
        difficulty_level=difficulty_level,
    )
