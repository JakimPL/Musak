from collections.abc import Iterable, Iterator
from pathlib import Path

from musak_model.data.config import (
    SegmentationConfig,
    SegmentationMode,
)
from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.data.scale_matcher.matcher import match_scale
from musak_model.data.schema import ParsedScore, Segment
from musak_model.data.segmenter.streams import tokenize_unified_stream_safely
from musak_model.data.segmenter.windows import create_window
from musak_model.data.tokenization_context import tokenization_context_from_scale_match
from musak_model.tokens.duration import DurationVocabulary
from musak_shared.profiling import NULL_PROFILER, ProfilerProtocol


def segment_score(
    score: ParsedScore,
    source_file: Path,
    *,
    duration_vocabulary: DurationVocabulary,
    segmentation: SegmentationConfig,
    scale_matcher_config: ScaleMatcherConfig,
    difficulty_level: int | None = None,
    profiler: ProfilerProtocol = NULL_PROFILER,
) -> list[Segment]:
    return list(
        iter_score_segments(
            score,
            source_file,
            duration_vocabulary=duration_vocabulary,
            segmentation=segmentation,
            scale_matcher_config=scale_matcher_config,
            difficulty_level=difficulty_level,
            profiler=profiler,
        )
    )


def iter_score_segments(
    score: ParsedScore,
    source_file: Path,
    *,
    duration_vocabulary: DurationVocabulary,
    segmentation: SegmentationConfig,
    scale_matcher_config: ScaleMatcherConfig,
    difficulty_level: int | None = None,
    profiler: ProfilerProtocol = NULL_PROFILER,
) -> Iterator[Segment]:
    total_bars = min(len(score.right_hand_bars), len(score.left_hand_bars))
    if segmentation.mode == SegmentationMode.WHOLE_FILE:
        yield from _segment_ranges(
            _whole_file_ranges(total_bars),
            score,
            source_file,
            duration_vocabulary=duration_vocabulary,
            segmentation=segmentation,
            scale_matcher_config=scale_matcher_config,
            difficulty_level=difficulty_level,
            profiler=profiler,
        )
        return

    yield from _segment_ranges(
        _window_ranges(total_bars, segmentation=segmentation),
        score,
        source_file,
        duration_vocabulary=duration_vocabulary,
        segmentation=segmentation,
        scale_matcher_config=scale_matcher_config,
        difficulty_level=difficulty_level,
        profiler=profiler,
    )


def _whole_file_ranges(total_bars: int) -> Iterator[tuple[int, int]]:
    if total_bars > 0:
        yield 0, total_bars


def _window_ranges(total_bars: int, *, segmentation: SegmentationConfig) -> Iterator[tuple[int, int]]:
    for start in range(0, total_bars - segmentation.window_bars + 1, segmentation.stride_bars):
        yield start, start + segmentation.window_bars


def _segment_ranges(
    ranges: Iterable[tuple[int, int]],
    score: ParsedScore,
    source_file: Path,
    *,
    duration_vocabulary: DurationVocabulary,
    segmentation: SegmentationConfig,
    scale_matcher_config: ScaleMatcherConfig,
    difficulty_level: int | None,
    profiler: ProfilerProtocol,
) -> Iterator[Segment]:
    for start, end in ranges:
        with profiler.measure("scale_match", source_file=source_file):
            scale_match = match_scale(
                score.right_hand_bars[start:end],
                score.left_hand_bars[start:end],
                config=scale_matcher_config,
            )
            tokenization_context = tokenization_context_from_scale_match(scale_match)
        with profiler.measure("score_copy", source_file=source_file):
            tokenization_score = score.model_copy(
                update={
                    "scale_root": scale_match.scale_root,
                    "key_fifths": tokenization_context.spelling_key_fifths,
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
            yield create_window(
                unified_window_bars=unified_window_bars,
                score=score,
                source_file=source_file,
                segmentation=segmentation,
                start=start,
                end=end,
                scale_match=scale_match,
                tokenization_context=tokenization_context,
                difficulty_level=difficulty_level,
            )
