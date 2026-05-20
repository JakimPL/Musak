from fractions import Fraction

from musak_model.data.schema import ParsedScore
from musak_model.data.segmenter.bar import bar_measure_duration, normalize_bar_events
from musak_model.data.segmenter.errors import TokenizationIneligibilityError
from musak_model.data.segmenter.types import BarTokenization, TieState, TimedTokenGroup
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    Token,
)

_JOIN_WITH_PREVIOUS_TOKEN: JoinWithPreviousToken = JoinWithPreviousToken()


def tokenize_unified_stream_safely(
    *,
    score: ParsedScore,
    duration_vocabulary: DurationVocabulary,
) -> list[BarTokenization]:
    total_bars = min(len(score.right_hand_bars), len(score.left_hand_bars))
    tokenized_bars: list[BarTokenization] = []
    right_tie_state: TieState | None = None
    left_tie_state: TieState | None = None

    for bar_index in range(total_bars):
        try:
            tokens, right_tie_state, left_tie_state = _tokenize_unified_bar(
                score=score,
                bar_index=bar_index,
                duration_vocabulary=duration_vocabulary,
                right_tie_state=right_tie_state,
                left_tie_state=left_tie_state,
            )
            tokenized_bars.append(BarTokenization(tokens=tokens))
        except TokenizationIneligibilityError as exception:
            right_tie_state = None
            left_tie_state = None
            tokenized_bars.append(
                BarTokenization(
                    tokens=[],
                    ineligibility_reasons=frozenset({exception.reason}),
                )
            )

    return tokenized_bars


def tokenize_unified_stream(
    *,
    score: ParsedScore,
    duration_vocabulary: DurationVocabulary,
) -> list[list[Token]]:
    total_bars = min(len(score.right_hand_bars), len(score.left_hand_bars))
    tokenized_bars: list[list[Token]] = []
    right_tie_state: TieState | None = None
    left_tie_state: TieState | None = None

    for bar_index in range(total_bars):
        tokens, right_tie_state, left_tie_state = _tokenize_unified_bar(
            score=score,
            bar_index=bar_index,
            duration_vocabulary=duration_vocabulary,
            right_tie_state=right_tie_state,
            left_tie_state=left_tie_state,
        )
        tokenized_bars.append(tokens)

    return tokenized_bars


def _tokenize_unified_bar(
    *,
    score: ParsedScore,
    bar_index: int,
    duration_vocabulary: DurationVocabulary,
    right_tie_state: TieState | None,
    left_tie_state: TieState | None,
) -> tuple[list[Token], TieState | None, TieState | None]:
    right_normalized = normalize_bar_events(
        bar=score.right_hand_bars[bar_index],
        bar_index=bar_index,
        hand=Hand.RIGHT,
        score=score,
        duration_vocabulary=duration_vocabulary,
        measure_duration=bar_measure_duration(score.right_hand_bars[bar_index]),
        tie_state=right_tie_state,
    )
    left_normalized = normalize_bar_events(
        bar=score.left_hand_bars[bar_index],
        bar_index=bar_index,
        hand=Hand.LEFT,
        score=score,
        duration_vocabulary=duration_vocabulary,
        measure_duration=bar_measure_duration(score.left_hand_bars[bar_index]),
        tie_state=left_tie_state,
    )

    return (
        merge_hand_groups(right_normalized.groups + left_normalized.groups),
        right_normalized.tie_state,
        left_normalized.tie_state,
    )


def merge_hand_groups(groups: list[TimedTokenGroup]) -> list[Token]:
    tokens: list[Token] = []
    active_hand: Hand | None = None
    previous_note_onset: tuple[int, Fraction] | None = None

    for group in sorted(groups, key=_group_sort_key):
        if active_hand != group.hand:
            tokens.append(HandToken(hand=group.hand))
            active_hand = group.hand

        for token in group.tokens:
            tokens.append(token)
            match token:
                case NoteToken():
                    current_onset = (group.bar_index, group.offset)
                    if previous_note_onset == current_onset:
                        tokens.append(_JOIN_WITH_PREVIOUS_TOKEN)
                    previous_note_onset = current_onset
                case HoldToken() | RestToken() | HandToken() | BarToken() | EndToken() | JoinWithPreviousToken():
                    continue

    return tokens


def _group_sort_key(group: TimedTokenGroup) -> tuple[Fraction, int, int]:
    return group.offset, _hand_sort_index(group.hand), _group_lowest_pitch(group)


def _hand_sort_index(hand: Hand) -> int:
    match hand:
        case Hand.RIGHT:
            return 0
        case Hand.LEFT:
            return 1


def _group_lowest_pitch(group: TimedTokenGroup) -> int:
    for index, token in enumerate(group.tokens):
        match token:
            case NoteToken():
                return index
            case HoldToken() | RestToken() | HandToken() | BarToken() | EndToken() | JoinWithPreviousToken():
                continue

    return -1
