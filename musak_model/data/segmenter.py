from pathlib import Path

from musak_model.data.config import SegmentationConfig
from musak_model.data.converter import pitch_to_degree
from musak_model.data.quantizer import quantize_duration_to_id
from musak_model.data.schema import (
    ParsedBar,
    ParsedChord,
    ParsedEvent,
    ParsedNote,
    ParsedRest,
    ParsedScore,
    Segment,
    SegmentMetadata,
)
from musak_model.tokens.duration_vocabulary import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    NoteToken,
    RestToken,
    ScaleType,
    Token,
)

_BAR_TOKEN: BarToken = BarToken()
_END_TOKEN: EndToken = EndToken()


def segment_score(
    score: ParsedScore,
    source_file: Path,
    *,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
    segmentation: SegmentationConfig,
    difficulty_level: int | None = None,
) -> list[Segment]:
    right_hand_tokens = _tokenize_hand(
        score.right_hand_bars,
        score=score,
        hand=Hand.RIGHT,
        scale_type=scale_type,
        duration_vocabulary=duration_vocabulary,
    )
    left_hand_tokens = _tokenize_hand(
        score.left_hand_bars,
        score=score,
        hand=Hand.LEFT,
        scale_type=scale_type,
        duration_vocabulary=duration_vocabulary,
    )

    return _create_windows(
        right_hand_tokens=right_hand_tokens,
        left_hand_tokens=left_hand_tokens,
        score=score,
        source_file=source_file,
        scale_type=scale_type,
        segmentation=segmentation,
        difficulty_level=difficulty_level,
    )


def _tokenize_hand(
    bars: list[ParsedBar],
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[list[Token]]:
    return [
        _tokenize_bar(
            bar,
            score=score,
            hand=hand,
            scale_type=scale_type,
            duration_vocabulary=duration_vocabulary,
        )
        for bar in bars
    ]


def _tokenize_bar(
    bar: ParsedBar,
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    tokens: list[Token] = []
    for event in bar.events:
        event_tokens = _tokenize_event(
            event,
            score=score,
            hand=hand,
            scale_type=scale_type,
            duration_vocabulary=duration_vocabulary,
        )
        tokens.extend(event_tokens)

    return tokens


def _tokenize_event(
    event: ParsedEvent,
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    if isinstance(event, ParsedNote):
        return [
            _note_to_token(
                event,
                score=score,
                hand=hand,
                scale_type=scale_type,
                duration_vocabulary=duration_vocabulary,
            )
        ]

    if isinstance(event, ParsedRest):
        return [
            RestToken(
                duration_id=quantize_duration_to_id(
                    event.duration,
                    vocabulary=duration_vocabulary,
                )
            )
        ]

    if isinstance(event, ParsedChord):
        return [
            _chord_to_token(
                event,
                score=score,
                hand=hand,
                scale_type=scale_type,
                duration_vocabulary=duration_vocabulary,
            )
        ]

    raise ValueError(f"unexpected event type: {type(event)}")


def _note_to_token(
    event: ParsedNote,
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> NoteToken:
    pitch_degree = pitch_to_degree(
        event.midi_pitch,
        key_root=score.key_root,
        key_fifths=score.key_fifths,
        scale_type=scale_type,
        hand=hand,
    )
    duration_id = quantize_duration_to_id(event.duration, vocabulary=duration_vocabulary)
    return NoteToken(
        degree=pitch_degree.degree,
        accidental=pitch_degree.accidental,
        octave_offset=pitch_degree.octave_offset,
        duration_id=duration_id,
    )


def _chord_to_token(
    event: ParsedChord,
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> NoteToken:
    top_pitch = max(event.midi_pitches) if hand == Hand.RIGHT else min(event.midi_pitches)
    pitch_degree = pitch_to_degree(
        top_pitch,
        key_root=score.key_root,
        key_fifths=score.key_fifths,
        scale_type=scale_type,
        hand=hand,
    )
    duration_id = quantize_duration_to_id(event.duration, vocabulary=duration_vocabulary)
    return NoteToken(
        degree=pitch_degree.degree,
        accidental=pitch_degree.accidental,
        octave_offset=pitch_degree.octave_offset,
        duration_id=duration_id,
    )


def _create_windows(
    right_hand_tokens: list[list[Token]],
    left_hand_tokens: list[list[Token]],
    score: ParsedScore,
    source_file: Path,
    *,
    scale_type: ScaleType,
    segmentation: SegmentationConfig,
    difficulty_level: int | None,
) -> list[Segment]:
    total_bars = min(len(right_hand_tokens), len(left_hand_tokens))
    segments: list[Segment] = []

    for start in range(0, total_bars - segmentation.window_bars + 1, segmentation.stride_bars):
        end = start + segmentation.window_bars
        right_window = _flatten_bars(right_hand_tokens[start:end])
        left_window = _flatten_bars(left_hand_tokens[start:end])

        segments.append(
            Segment(
                right_hand_tokens=right_window,
                left_hand_tokens=left_window,
                metadata=SegmentMetadata(
                    key_root=score.key_root,
                    scale_type=scale_type,
                    time_numerator=score.time_numerator,
                    time_denominator=score.time_denominator,
                    bar_count=segmentation.window_bars,
                    window_start_bar=start,
                    source_file=source_file,
                    difficulty_level=difficulty_level,
                ),
            )
        )

    return segments


def _flatten_bars(bar_token_lists: list[list[Token]]) -> list[Token]:
    tokens: list[Token] = []
    for bar_tokens in bar_token_lists:
        tokens.extend(bar_tokens)
        tokens.append(_BAR_TOKEN)

    tokens.append(_END_TOKEN)
    return tokens
