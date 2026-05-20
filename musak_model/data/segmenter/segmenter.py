from pathlib import Path

from musak_model.data.config import (
    DEFAULT_SCALE_MATCH_MINIMUM_BEST_MARGIN,
    DEFAULT_SCALE_MATCH_MINIMUM_IN_SCALE_WEIGHT_FRACTION,
    SegmentationConfig,
)
from musak_model.data.scale_match import match_scale
from musak_model.data.schema import ParsedScore, Segment
from musak_model.data.segmenter.streams import tokenize_unified_stream_safely
from musak_model.data.segmenter.windows import create_window
from musak_model.tokens.duration import DurationVocabulary


def segment_score(
    score: ParsedScore,
    source_file: Path,
    *,
    duration_vocabulary: DurationVocabulary,
    segmentation: SegmentationConfig,
    difficulty_level: int | None = None,
    scale_match_minimum_in_scale_weight_fraction: float = DEFAULT_SCALE_MATCH_MINIMUM_IN_SCALE_WEIGHT_FRACTION,
    scale_match_minimum_best_margin: float = DEFAULT_SCALE_MATCH_MINIMUM_BEST_MARGIN,
) -> list[Segment]:
    total_bars = min(len(score.right_hand_bars), len(score.left_hand_bars))
    segments: list[Segment] = []
    for start in range(0, total_bars - segmentation.window_bars + 1, segmentation.stride_bars):
        end = start + segmentation.window_bars
        scale_match = match_scale(
            score.right_hand_bars[start:end],
            score.left_hand_bars[start:end],
            minimum_in_scale_weight_fraction=scale_match_minimum_in_scale_weight_fraction,
            minimum_best_margin=scale_match_minimum_best_margin,
        )
        tokenization_score = score.model_copy(
            update={
                "scale_root": scale_match.scale_root,
                "key_fifths": (
                    scale_match.diagnostics.declared_key_fifths
                    if scale_match.diagnostics.declared_key_fifths is not None
                    else 0
                ),
                "scale_type": scale_match.scale_type,
                "right_hand_bars": score.right_hand_bars[:end],
                "left_hand_bars": score.left_hand_bars[:end],
            }
        )
        unified_window_bars = tokenize_unified_stream_safely(
            score=tokenization_score,
            duration_vocabulary=duration_vocabulary,
        )[start:end]
        segments.append(
            create_window(
                unified_window_bars=unified_window_bars,
                score=score,
                source_file=source_file,
                segmentation=segmentation,
                start=start,
                end=end,
                scale_match=scale_match,
                difficulty_level=difficulty_level,
            )
        )

    return segments
