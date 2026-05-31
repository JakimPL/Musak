from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Final, Sequence

from musak_model.generation.constraints import GenerationConstraints
from musak_model.tokens.duration import DurationVocabulary, duration_fraction_to_ticks
from musak_model.tokens.factorized import hand_to_attribute_id
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    StartToken,
    Token,
)
from musak_model.tokens.vocabulary import TokenVocabulary

NO_DECODER_COORDINATE_ID: Final = -1
MINIMUM_BAR_DURATION_TICKS: Final = 1


@dataclass(frozen=True)
class DecoderInputCoordinates:
    bar_relative_ticks: tuple[int, ...]
    bar_duration_ticks: tuple[int, ...]
    active_hand_ids: tuple[int, ...]


@dataclass(frozen=True)
class _OnsetCursor:
    start: Fraction
    duration: Fraction


@dataclass(frozen=True)
class _PendingJoinCursor:
    hand: Hand
    target: _OnsetCursor
    cursor_before_note: Fraction
    duration: Fraction


@dataclass(frozen=True)
class _DecoderCoordinateState:
    constraints: GenerationConstraints
    active_hand: Hand = Hand.RIGHT
    bar_index: int = 0
    right_cursor: Fraction = Fraction(0)
    left_cursor: Fraction = Fraction(0)
    right_last_onset: _OnsetCursor | None = None
    left_last_onset: _OnsetCursor | None = None
    pending_join: _PendingJoinCursor | None = None

    def apply(self, token: Token, *, duration_vocabulary: DurationVocabulary) -> _DecoderCoordinateState:
        match token:
            case StartToken() | EndToken():
                return self
            case HandToken():
                return replace(self, active_hand=token.hand, pending_join=None)
            case NoteToken():
                return self._apply_note(token, duration_vocabulary=duration_vocabulary)
            case RestToken():
                return self._apply_duration_token(token.duration_id, duration_vocabulary=duration_vocabulary)
            case HoldToken():
                return self._apply_duration_token(token.duration_id, duration_vocabulary=duration_vocabulary)
            case JoinWithPreviousToken():
                return self._apply_join()
            case BarToken():
                return self._apply_bar()

        raise ValueError(f"unsupported token type: {type(token)}")

    def cursor(self, hand: Hand) -> Fraction:
        return self.right_cursor if hand == Hand.RIGHT else self.left_cursor

    def with_cursor(self, hand: Hand, cursor: Fraction) -> _DecoderCoordinateState:
        if hand == Hand.RIGHT:
            return replace(self, right_cursor=cursor)

        return replace(self, left_cursor=cursor)

    def last_onset(self, hand: Hand) -> _OnsetCursor | None:
        return self.right_last_onset if hand == Hand.RIGHT else self.left_last_onset

    def with_last_onset(self, hand: Hand, onset: _OnsetCursor) -> _DecoderCoordinateState:
        if hand == Hand.RIGHT:
            return replace(self, right_last_onset=onset)

        return replace(self, left_last_onset=onset)

    def _apply_note(
        self,
        token: NoteToken,
        *,
        duration_vocabulary: DurationVocabulary,
    ) -> _DecoderCoordinateState:
        duration = duration_vocabulary.id_to_fraction(token.duration_id)
        cursor_before_note = self.cursor(self.active_hand)
        join_target = self._same_hand_join_target(duration)
        state = self.with_cursor(self.active_hand, cursor_before_note + duration).with_last_onset(
            self.active_hand,
            _OnsetCursor(start=cursor_before_note, duration=duration),
        )
        pending_join = (
            _PendingJoinCursor(
                hand=self.active_hand,
                target=join_target,
                cursor_before_note=cursor_before_note,
                duration=duration,
            )
            if join_target is not None
            else None
        )
        return replace(state, pending_join=pending_join)

    def _apply_duration_token(
        self,
        duration_id: int,
        *,
        duration_vocabulary: DurationVocabulary,
    ) -> _DecoderCoordinateState:
        duration = duration_vocabulary.id_to_fraction(duration_id)
        return replace(
            self.with_cursor(self.active_hand, self.cursor(self.active_hand) + duration),
            pending_join=None,
        )

    def _apply_join(self) -> _DecoderCoordinateState:
        pending_join = self.pending_join
        if pending_join is None or pending_join.hand != self.active_hand:
            return replace(self, pending_join=None)

        joined_cursor = max(pending_join.cursor_before_note, pending_join.target.start + pending_join.duration)
        state = self.with_cursor(self.active_hand, joined_cursor).with_last_onset(
            self.active_hand,
            pending_join.target,
        )
        return replace(state, pending_join=None)

    def _apply_bar(self) -> _DecoderCoordinateState:
        next_bar_start = self.constraints.bar_end(self.bar_index)
        return replace(
            self,
            bar_index=self.bar_index + 1,
            right_cursor=max(self.right_cursor, next_bar_start),
            left_cursor=max(self.left_cursor, next_bar_start),
            pending_join=None,
        )

    def _same_hand_join_target(self, duration: Fraction) -> _OnsetCursor | None:
        target = self.last_onset(self.active_hand)
        if target is None:
            return None

        if target.duration != duration:
            return None

        if target.start + target.duration != self.cursor(self.active_hand):
            return None

        return target


def decoder_input_coordinates_from_token_ids(
    prefix_token_ids: Sequence[int],
    *,
    constraints: GenerationConstraints,
    token_vocabulary: TokenVocabulary,
    duration_vocabulary: DurationVocabulary,
    duration_tick_denominator: int,
) -> DecoderInputCoordinates:
    return decoder_input_coordinates_from_tokens(
        token_vocabulary.decode(list(prefix_token_ids)),
        constraints=constraints,
        duration_vocabulary=duration_vocabulary,
        duration_tick_denominator=duration_tick_denominator,
    )


def decoder_input_coordinates_from_tokens(
    prefix_tokens: Sequence[Token],
    *,
    constraints: GenerationConstraints,
    duration_vocabulary: DurationVocabulary,
    duration_tick_denominator: int,
) -> DecoderInputCoordinates:
    state = _DecoderCoordinateState(constraints=constraints)
    bar_relative_ticks: list[int] = []
    bar_duration_ticks: list[int] = []
    active_hand_ids: list[int] = []
    _append_state_coordinates(
        state,
        bar_relative_ticks=bar_relative_ticks,
        bar_duration_ticks=bar_duration_ticks,
        active_hand_ids=active_hand_ids,
        duration_tick_denominator=duration_tick_denominator,
    )
    for token in prefix_tokens:
        state = state.apply(token, duration_vocabulary=duration_vocabulary)
        _append_state_coordinates(
            state,
            bar_relative_ticks=bar_relative_ticks,
            bar_duration_ticks=bar_duration_ticks,
            active_hand_ids=active_hand_ids,
            duration_tick_denominator=duration_tick_denominator,
        )

    return DecoderInputCoordinates(
        bar_relative_ticks=tuple(bar_relative_ticks),
        bar_duration_ticks=tuple(bar_duration_ticks),
        active_hand_ids=tuple(active_hand_ids),
    )


def _append_state_coordinates(
    state: _DecoderCoordinateState,
    *,
    bar_relative_ticks: list[int],
    bar_duration_ticks: list[int],
    active_hand_ids: list[int],
    duration_tick_denominator: int,
) -> None:
    bar_start = state.constraints.bar_start(state.bar_index)
    active_cursor = state.cursor(state.active_hand)
    bar_duration = state.constraints.bar_duration(state.bar_index)
    bar_relative_ticks.append(
        duration_fraction_to_ticks(active_cursor - bar_start, denominator=duration_tick_denominator)
    )
    bar_duration_ticks.append(
        max(
            MINIMUM_BAR_DURATION_TICKS,
            duration_fraction_to_ticks(bar_duration, denominator=duration_tick_denominator),
        )
    )
    active_hand_ids.append(hand_to_attribute_id(state.active_hand))
