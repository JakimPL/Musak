from pathlib import Path

from musak_model.data.config import (
    DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT,
    DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION,
    DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN,
    DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN,
    SegmentationConfig,
    SegmentationMode,
)
from musak_model.data.scale_match import match_scale
from musak_model.data.schema import ParsedScore, Segment
from musak_model.data.segmenter.streams import tokenize_unified_stream_safely
from musak_model.data.segmenter.windows import create_window
from musak_model.processing.profiler import NULL_PROCESSING_PROFILER, ProcessingProfilerProtocol
from musak_model.tokens.duration import DurationVocabulary


def segment_score(
    score: ParsedScore,
    source_file: Path,
    *,
    duration_vocabulary: DurationVocabulary,
    segmentation: SegmentationConfig,
    difficulty_level: int | None = None,
    scale_match_support_score_margin: float = DEFAULT_SCALE_MATCH_SUPPORT_SCORE_MARGIN,
    scale_match_selection_score_margin: float = DEFAULT_SCALE_MATCH_SELECTION_SCORE_MARGIN,
    scale_match_maximum_unexplained_weight_fraction: float = DEFAULT_SCALE_MATCH_MAXIMUM_UNEXPLAINED_WEIGHT_FRACTION,
    scale_match_maximum_explanation_pitch_class_count: int = DEFAULT_SCALE_MATCH_MAXIMUM_EXPLANATION_PITCH_CLASS_COUNT,
    profiler: ProcessingProfilerProtocol = NULL_PROCESSING_PROFILER,
) -> list[Segment]:
    total_bars = min(len(score.right_hand_bars), len(score.left_hand_bars))
    if segmentation.mode == SegmentationMode.WHOLE_FILE:
        return _segment_ranges(
            [(0, total_bars)] if total_bars > 0 else [],
            score,
            source_file,
            duration_vocabulary=duration_vocabulary,
            segmentation=segmentation,
            difficulty_level=difficulty_level,
            scale_match_support_score_margin=scale_match_support_score_margin,
            scale_match_selection_score_margin=scale_match_selection_score_margin,
            scale_match_maximum_unexplained_weight_fraction=scale_match_maximum_unexplained_weight_fraction,
            scale_match_maximum_explanation_pitch_class_count=scale_match_maximum_explanation_pitch_class_count,
            profiler=profiler,
        )

    ranges = [
        (start, start + segmentation.window_bars)
        for start in range(0, total_bars - segmentation.window_bars + 1, segmentation.stride_bars)
    ]
    return _segment_ranges(
        ranges,
        score,
        source_file,
        duration_vocabulary=duration_vocabulary,
        segmentation=segmentation,
        difficulty_level=difficulty_level,
        scale_match_support_score_margin=scale_match_support_score_margin,
        scale_match_selection_score_margin=scale_match_selection_score_margin,
        scale_match_maximum_unexplained_weight_fraction=scale_match_maximum_unexplained_weight_fraction,
        scale_match_maximum_explanation_pitch_class_count=scale_match_maximum_explanation_pitch_class_count,
        profiler=profiler,
    )


def _segment_ranges(
    ranges: list[tuple[int, int]],
    score: ParsedScore,
    source_file: Path,
    *,
    duration_vocabulary: DurationVocabulary,
    segmentation: SegmentationConfig,
    difficulty_level: int | None,
    scale_match_support_score_margin: float,
    scale_match_selection_score_margin: float,
    scale_match_maximum_unexplained_weight_fraction: float,
    scale_match_maximum_explanation_pitch_class_count: int,
    profiler: ProcessingProfilerProtocol,
) -> list[Segment]:
    segments: list[Segment] = []
    for start, end in ranges:
        with profiler.measure("scale_match", source_file=source_file):
            scale_match = match_scale(
                score.right_hand_bars[start:end],
                score.left_hand_bars[start:end],
                support_score_margin=scale_match_support_score_margin,
                selection_score_margin=scale_match_selection_score_margin,
                maximum_unexplained_weight_fraction=scale_match_maximum_unexplained_weight_fraction,
                maximum_explanation_pitch_class_count=scale_match_maximum_explanation_pitch_class_count,
            )
        with profiler.measure("score_copy", source_file=source_file):
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
        with profiler.measure("tokenize_unified_stream", source_file=source_file):
            unified_window_bars = tokenize_unified_stream_safely(
                score=tokenization_score,
                duration_vocabulary=duration_vocabulary,
            )[start:end]
        with profiler.measure("create_window", source_file=source_file):
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
