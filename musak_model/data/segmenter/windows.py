from fractions import Fraction
from pathlib import Path

from musak_model.data.cleaning import is_silent_bar_pair
from musak_model.data.config import SegmentationConfig
from musak_model.data.scale_matcher.schema import ScaleMatch
from musak_model.data.schema import (
    ParsedScore,
    Segment,
    SegmentIneligibilityReason,
    SegmentMetadata,
    TokenizationContext,
)
from musak_model.data.segmenter.bar import paired_bar_measure_duration
from musak_model.data.segmenter.types import BarTokenization
from musak_model.data.tokenization_context import (
    tokenization_context_from_scale_match,
    tokenization_context_from_score,
)
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    NoteToken,
    Token,
)

_BAR_TOKEN: BarToken = BarToken()
_END_TOKEN: EndToken = EndToken()


def create_windows(
    unified_tokens: list[BarTokenization],
    score: ParsedScore,
    source_file: Path,
    *,
    segmentation: SegmentationConfig,
    difficulty_level: int | None = None,
) -> list[Segment]:
    total_bars = len(unified_tokens)
    segments: list[Segment] = []

    for start in range(0, total_bars - segmentation.window_bars + 1, segmentation.stride_bars):
        end = start + segmentation.window_bars
        unified_window_bars = unified_tokens[start:end]
        segments.append(
            create_window(
                unified_window_bars=unified_window_bars,
                score=score,
                source_file=source_file,
                segmentation=segmentation,
                start=start,
                end=end,
                scale_match=None,
                tokenization_context=None,
                difficulty_level=difficulty_level,
            )
        )

    return segments


def create_window(
    *,
    unified_window_bars: list[BarTokenization],
    score: ParsedScore,
    source_file: Path,
    segmentation: SegmentationConfig,
    start: int,
    end: int,
    scale_match: ScaleMatch | None,
    tokenization_context: TokenizationContext | None = None,
    difficulty_level: int | None = None,
) -> Segment:
    unified_window = _flatten_bars([bar.tokens for bar in unified_window_bars])
    first_bar = score.right_hand_bars[start]
    resolved_tokenization_context = _resolve_tokenization_context(
        score=score,
        scale_match=scale_match,
        tokenization_context=tokenization_context,
    )
    ineligibility_reasons = _merge_ineligibility_reasons(
        _segment_ineligibility_reasons(
            score=score,
            start=start,
            end=end,
        ),
        _scale_match_ineligibility_reasons(scale_match),
        _window_token_ineligibility_reasons(
            first_bar_tokens=unified_window_bars[0].tokens,
            window_start_bar=start,
        ),
        *(bar.ineligibility_reasons for bar in unified_window_bars),
    )
    return Segment(
        tokens=unified_window,
        metadata=SegmentMetadata(
            scale_root=scale_match.scale_root if scale_match is not None else score.scale_root,
            scale_type=scale_match.scale_type if scale_match is not None else score.scale_type,
            tokenization_context=resolved_tokenization_context,
            time_numerator=first_bar.time_numerator,
            time_denominator=first_bar.time_denominator,
            bar_count=end - start,
            bar_durations=_bar_durations(score=score, start=start, end=end),
            window_start_bar=start,
            source_file=source_file,
            difficulty_level=difficulty_level,
            difficulty_features=None,
            scale_match=scale_match.diagnostics if scale_match is not None else None,
            eligible_for_training=not ineligibility_reasons,
            ineligibility_reasons=ineligibility_reasons,
        ),
    )


def _resolve_tokenization_context(
    *,
    score: ParsedScore,
    scale_match: ScaleMatch | None,
    tokenization_context: TokenizationContext | None,
) -> TokenizationContext:
    if tokenization_context is not None:
        return tokenization_context

    if scale_match is not None:
        return tokenization_context_from_scale_match(scale_match)

    return tokenization_context_from_score(score)


def _bar_durations(
    *,
    score: ParsedScore,
    start: int,
    end: int,
) -> tuple[Fraction, ...]:
    return tuple(
        paired_bar_measure_duration(score.right_hand_bars[index], score.left_hand_bars[index])
        for index in range(start, end)
    )


def _segment_ineligibility_reasons(
    *,
    score: ParsedScore,
    start: int,
    end: int,
) -> frozenset[SegmentIneligibilityReason]:
    first_bar = score.right_hand_bars[start]
    first_time_signature = (first_bar.time_numerator, first_bar.time_denominator)
    first_key_fifths = first_bar.declared_key_fifths
    reasons: set[SegmentIneligibilityReason] = set()

    window_bars = score.right_hand_bars[start:end] + score.left_hand_bars[start:end]
    if any((bar.time_numerator, bar.time_denominator) != first_time_signature for bar in window_bars):
        reasons.add(SegmentIneligibilityReason.MIXED_TIME_SIGNATURE)

    if any(bar.declared_key_fifths != first_key_fifths for bar in window_bars):
        reasons.add(SegmentIneligibilityReason.KEY_SIGNATURE_CHANGE)

    if _has_silent_edge_bar(score=score, start=start, end=end):
        reasons.add(SegmentIneligibilityReason.SILENT_EDGE_BAR)

    return frozenset(reasons)


def _scale_match_ineligibility_reasons(scale_match: ScaleMatch | None) -> frozenset[SegmentIneligibilityReason]:
    if scale_match is None:
        return frozenset()

    reasons: set[SegmentIneligibilityReason] = set()
    if scale_match.diagnostics.no_pitches:
        reasons.add(SegmentIneligibilityReason.SCALE_MATCH_NO_PITCHES)
    if scale_match.diagnostics.low_confidence:
        reasons.add(SegmentIneligibilityReason.SCALE_MATCH_LOW_CONFIDENCE)

    return frozenset(reasons)


def _has_silent_edge_bar(
    *,
    score: ParsedScore,
    start: int,
    end: int,
) -> bool:
    first_index = start
    last_index = end - 1
    return is_silent_bar_pair(
        score.right_hand_bars[first_index],
        score.left_hand_bars[first_index],
    ) or is_silent_bar_pair(
        score.right_hand_bars[last_index],
        score.left_hand_bars[last_index],
    )


def _window_token_ineligibility_reasons(
    *,
    first_bar_tokens: list[Token],
    window_start_bar: int,
) -> frozenset[SegmentIneligibilityReason]:
    if window_start_bar == 0:
        return frozenset()

    active_hand = Hand.RIGHT
    seen_same_hand_attack = {Hand.RIGHT: False, Hand.LEFT: False}
    for token in first_bar_tokens:
        if isinstance(token, HandToken):
            active_hand = token.hand
            continue

        if isinstance(token, HoldToken) and not seen_same_hand_attack[active_hand]:
            return frozenset({SegmentIneligibilityReason.TIE_CONTINUATION_AT_WINDOW_START})

        if isinstance(token, NoteToken):
            seen_same_hand_attack[active_hand] = True

    return frozenset()


def _merge_ineligibility_reasons(
    *reason_groups: frozenset[SegmentIneligibilityReason],
) -> frozenset[SegmentIneligibilityReason]:
    return frozenset().union(*reason_groups)


def _flatten_bars(bar_token_lists: list[list[Token]]) -> list[Token]:
    tokens: list[Token] = []
    for bar_tokens in bar_token_lists:
        tokens.extend(bar_tokens)
        tokens.append(_BAR_TOKEN)

    tokens.append(_END_TOKEN)
    return tokens
