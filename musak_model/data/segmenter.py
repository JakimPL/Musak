from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

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
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    ScaleType,
    Token,
)

_BAR_TOKEN: BarToken = BarToken()
_END_TOKEN: EndToken = EndToken()
_JOIN_WITH_PREVIOUS_TOKEN: JoinWithPreviousToken = JoinWithPreviousToken()


class _TimedTokenGroup(NamedTuple):
    bar_index: int
    offset: Fraction
    hand: Hand
    tokens: list[Token]


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
    unified_tokens = _tokenize_unified_stream(
        score=score,
        scale_type=scale_type,
        duration_vocabulary=duration_vocabulary,
    )

    return _create_windows(
        right_hand_tokens=right_hand_tokens,
        left_hand_tokens=left_hand_tokens,
        unified_tokens=unified_tokens,
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
    measure_duration = Fraction(score.time_numerator, score.time_denominator)
    return [
        _tokens_from_bar_groups(
            _normalize_bar_events(
                bar=bar,
                bar_index=index,
                hand=hand,
                score=score,
                scale_type=scale_type,
                duration_vocabulary=duration_vocabulary,
                measure_duration=measure_duration,
            )
        )
        for index, bar in enumerate(bars)
    ]


def _tokenize_unified_stream(
    *,
    score: ParsedScore,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[list[Token]]:
    measure_duration = Fraction(score.time_numerator, score.time_denominator)
    total_bars = min(len(score.right_hand_bars), len(score.left_hand_bars))
    tokenized_bars: list[list[Token]] = []

    for bar_index in range(total_bars):
        right_groups = _normalize_bar_events(
            bar=score.right_hand_bars[bar_index],
            bar_index=bar_index,
            hand=Hand.RIGHT,
            score=score,
            scale_type=scale_type,
            duration_vocabulary=duration_vocabulary,
            measure_duration=measure_duration,
        )
        left_groups = _normalize_bar_events(
            bar=score.left_hand_bars[bar_index],
            bar_index=bar_index,
            hand=Hand.LEFT,
            score=score,
            scale_type=scale_type,
            duration_vocabulary=duration_vocabulary,
            measure_duration=measure_duration,
        )
        tokenized_bars.append(_merge_hand_groups(right_groups + left_groups))

    return tokenized_bars


def _normalize_bar_events(
    *,
    bar: ParsedBar,
    bar_index: int,
    hand: Hand,
    score: ParsedScore,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
    measure_duration: Fraction,
) -> list[_TimedTokenGroup]:
    cursor = Fraction(0, 1)
    groups: list[_TimedTokenGroup] = []

    for event in sorted(bar.events, key=_event_sort_key):
        if event.beat_offset < cursor:
            raise ValueError(
                f"overlapping events in {hand.value} hand at bar {bar_index}: "
                f"event starts at {event.beat_offset}, previous event ends at {cursor}"
            )

        if event.beat_offset > cursor:
            rest_duration = event.beat_offset - cursor
            groups.append(
                _TimedTokenGroup(
                    bar_index=bar_index,
                    offset=cursor,
                    hand=hand,
                    tokens=[
                        RestToken(duration_id=quantize_duration_to_id(rest_duration, vocabulary=duration_vocabulary))
                    ],
                )
            )

        groups.append(
            _TimedTokenGroup(
                bar_index=bar_index,
                offset=event.beat_offset,
                hand=hand,
                tokens=_tokenize_event(
                    event,
                    score=score,
                    hand=hand,
                    scale_type=scale_type,
                    duration_vocabulary=duration_vocabulary,
                ),
            )
        )
        cursor = event.beat_offset + event.duration

    if cursor > measure_duration:
        raise ValueError(
            f"{hand.value} hand bar {bar_index} exceeds measure duration {measure_duration}: ends at {cursor}"
        )

    if cursor < measure_duration:
        groups.append(
            _TimedTokenGroup(
                bar_index=bar_index,
                offset=cursor,
                hand=hand,
                tokens=[
                    RestToken(
                        duration_id=quantize_duration_to_id(measure_duration - cursor, vocabulary=duration_vocabulary)
                    )
                ],
            )
        )

    return groups


def _event_sort_key(event: ParsedEvent) -> tuple[Fraction, int]:
    return event.beat_offset, _lowest_pitch(event)


def _lowest_pitch(event: ParsedEvent) -> int:
    if isinstance(event, ParsedNote):
        return event.midi_pitch

    if isinstance(event, ParsedChord):
        return min(event.midi_pitches)

    return -1


def _tokens_from_bar_groups(groups: list[_TimedTokenGroup]) -> list[Token]:
    tokens: list[Token] = []
    for group in groups:
        tokens.extend(group.tokens)

    return tokens


def _merge_hand_groups(groups: list[_TimedTokenGroup]) -> list[Token]:
    tokens: list[Token] = []
    active_hand: Hand | None = None
    previous_note_onset: tuple[int, Fraction] | None = None

    for group in sorted(groups, key=_group_sort_key):
        if active_hand != group.hand:
            tokens.append(HandToken(hand=group.hand))
            active_hand = group.hand

        for token in group.tokens:
            tokens.append(token)
            if isinstance(token, NoteToken):
                current_onset = (group.bar_index, group.offset)
                if previous_note_onset == current_onset:
                    tokens.append(_JOIN_WITH_PREVIOUS_TOKEN)
                previous_note_onset = current_onset

    return tokens


def _group_sort_key(group: _TimedTokenGroup) -> tuple[Fraction, int, int]:
    return group.offset, _hand_sort_index(group.hand), _group_lowest_pitch(group)


def _hand_sort_index(hand: Hand) -> int:
    return 0 if hand == Hand.RIGHT else 1


def _group_lowest_pitch(group: _TimedTokenGroup) -> int:
    note_tokens = [index for index, token in enumerate(group.tokens) if isinstance(token, NoteToken)]
    return note_tokens[0] if note_tokens else -1


def _tokenize_bar(
    bar: ParsedBar,
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    return _tokens_from_bar_groups(
        _normalize_bar_events(
            bar=bar,
            bar_index=0,
            score=score,
            hand=hand,
            scale_type=scale_type,
            duration_vocabulary=duration_vocabulary,
            measure_duration=Fraction(score.time_numerator, score.time_denominator),
        )
    )


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
        return _chord_to_tokens(
            event,
            score=score,
            hand=hand,
            scale_type=scale_type,
            duration_vocabulary=duration_vocabulary,
        )

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


def _chord_to_tokens(
    event: ParsedChord,
    *,
    score: ParsedScore,
    hand: Hand,
    scale_type: ScaleType,
    duration_vocabulary: DurationVocabulary,
) -> list[Token]:
    sorted_pitches = sorted(event.midi_pitches)
    tokens: list[Token] = []
    for midi_pitch in sorted_pitches:
        note_token = _note_to_token(
            ParsedNote(midi_pitch=midi_pitch, duration=event.duration, beat_offset=event.beat_offset),
            score=score,
            hand=hand,
            scale_type=scale_type,
            duration_vocabulary=duration_vocabulary,
        )
        tokens.append(note_token)

    return tokens


def _create_windows(
    right_hand_tokens: list[list[Token]],
    left_hand_tokens: list[list[Token]],
    unified_tokens: list[list[Token]],
    score: ParsedScore,
    source_file: Path,
    *,
    scale_type: ScaleType,
    segmentation: SegmentationConfig,
    difficulty_level: int | None,
) -> list[Segment]:
    total_bars = min(len(right_hand_tokens), len(left_hand_tokens), len(unified_tokens))
    segments: list[Segment] = []

    for start in range(0, total_bars - segmentation.window_bars + 1, segmentation.stride_bars):
        end = start + segmentation.window_bars
        right_window = _flatten_bars(right_hand_tokens[start:end])
        left_window = _flatten_bars(left_hand_tokens[start:end])
        unified_window = _flatten_bars(unified_tokens[start:end])

        segments.append(
            Segment(
                tokens=unified_window,
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
