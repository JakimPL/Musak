from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Collection, Sequence

import torch
from torch import Tensor

from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.pitch import note_token_to_midi_pitch, note_token_to_static_hand_position
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    ScaleType,
    StartToken,
    Token,
)
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_shared.elements import is_dotted_duration


class GenerationConstraintError(ValueError):
    """Raised when a generated prefix violates hard generation constraints."""


@dataclass(frozen=True)
class GenerationConstraints:
    time_numerator: int
    time_denominator: int
    bar_count: int
    minimum_duration: Fraction | None = None
    allow_dotted_durations: bool = True
    max_notes_per_onset_per_hand: int | None = None
    maximum_pitch_gap_semitones: int | None = None
    maximum_static_hand_span_degrees: int | None = None
    key_root: int | None = None
    scale_type: ScaleType | None = None

    @property
    def measure_duration(self) -> Fraction:
        return Fraction(self.time_numerator, self.time_denominator)


@dataclass(frozen=True)
class _OnsetState:
    start: Fraction
    duration: Fraction
    note_count: int
    midi_pitches: tuple[int, ...]


@dataclass(frozen=True)
class _PendingJoin:
    hand: Hand
    target: _OnsetState
    cursor_before_note: Fraction
    cursor_after_note: Fraction
    duration: Fraction
    midi_pitch: int | None


@dataclass(frozen=True)
class GenerationConstraintState:
    constraints: GenerationConstraints
    active_hand: Hand = Hand.RIGHT
    bar_index: int = 0
    right_cursor: Fraction = Fraction(0)
    left_cursor: Fraction = Fraction(0)
    right_last_attack_end: Fraction | None = None
    left_last_attack_end: Fraction | None = None
    right_last_onset: _OnsetState | None = None
    left_last_onset: _OnsetState | None = None
    right_static_positions: tuple[int, ...] = ()
    left_static_positions: tuple[int, ...] = ()
    pending_join: _PendingJoin | None = None
    must_join: bool = False
    ended: bool = False

    def apply(
        self,
        token: Token,
        *,
        duration_vocabulary: DurationVocabulary,
    ) -> GenerationConstraintState:
        if self.ended:
            raise GenerationConstraintError("cannot emit tokens after EndToken")

        match token:
            case StartToken():
                return self
            case JoinWithPreviousToken():
                return self._apply_join()
            case _ if self.must_join:
                raise GenerationConstraintError("overflow chord note must be followed by JoinWithPreviousToken")
            case HandToken():
                return self._apply_hand(token)
            case NoteToken():
                return self._apply_note(token, duration_vocabulary=duration_vocabulary)
            case RestToken():
                return self._apply_rest(token, duration_vocabulary=duration_vocabulary)
            case HoldToken():
                return self._apply_hold(token, duration_vocabulary=duration_vocabulary)
            case BarToken():
                return self._apply_bar()
            case EndToken():
                return self._apply_end()

        raise GenerationConstraintError(f"unsupported token type: {type(token)}")

    def allowed_token_ids(
        self,
        *,
        token_vocabulary: TokenVocabulary,
        duration_vocabulary: DurationVocabulary,
    ) -> frozenset[int]:
        return frozenset(
            token_id
            for token_id in range(token_vocabulary.vocabulary_size)
            if self.allows(token_vocabulary.id_to_token(token_id), duration_vocabulary=duration_vocabulary)
        )

    def allows(self, token: Token, *, duration_vocabulary: DurationVocabulary) -> bool:
        match token:
            case StartToken():
                return False

        try:
            self.apply(token, duration_vocabulary=duration_vocabulary)
        except GenerationConstraintError:
            return False

        return True

    def _apply_hand(self, token: HandToken) -> GenerationConstraintState:
        self._raise_if_complete_bar_count(token_name="HandToken")
        return replace(self, active_hand=token.hand, pending_join=None)

    def _apply_note(
        self,
        token: NoteToken,
        *,
        duration_vocabulary: DurationVocabulary,
    ) -> GenerationConstraintState:
        self._raise_if_complete_bar_count(token_name="NoteToken")
        duration = self._checked_duration(token.duration_id, duration_vocabulary=duration_vocabulary)
        remaining = self._remaining_duration(self.active_hand)
        join_target = self._join_target(duration)
        can_join = join_target is not None
        midi_pitch = self._note_midi_pitch(token)
        exceeds_pitch_gap = self._exceeds_pitch_gap(midi_pitch)
        static_position = note_token_to_static_hand_position(token)
        if duration > remaining and not can_join:
            raise GenerationConstraintError("note duration exceeds remaining active-hand measure time")

        if exceeds_pitch_gap and not can_join:
            raise GenerationConstraintError("note exceeds maximum same-hand pitch gap")

        if self._exceeds_static_hand_span(static_position):
            raise GenerationConstraintError("note exceeds maximum static hand span")

        cursor = self._cursor(self.active_hand)
        cursor_after = cursor + duration
        pending_join = (
            _PendingJoin(
                hand=self.active_hand,
                target=join_target,
                cursor_before_note=cursor,
                cursor_after_note=cursor_after,
                duration=duration,
                midi_pitch=midi_pitch,
            )
            if join_target is not None
            else None
        )

        state = self.with_cursor(self.active_hand, cursor_after)
        state = state.with_last_attack_end(self.active_hand, cursor_after)
        state = state.with_static_position(self.active_hand, static_position)
        state = state.with_last_onset(
            self.active_hand,
            _OnsetState(
                start=cursor,
                duration=duration,
                note_count=1,
                midi_pitches=_optional_pitch_tuple(midi_pitch),
            ),
        )
        return replace(state, pending_join=pending_join, must_join=duration > remaining or exceeds_pitch_gap)

    def _apply_rest(
        self,
        token: RestToken,
        *,
        duration_vocabulary: DurationVocabulary,
    ) -> GenerationConstraintState:
        self._raise_if_complete_bar_count(token_name="RestToken")
        duration = self._checked_duration(token.duration_id, duration_vocabulary=duration_vocabulary)
        if duration > self._remaining_duration(self.active_hand):
            raise GenerationConstraintError("rest duration exceeds remaining active-hand measure time")

        return replace(
            self.with_cursor(self.active_hand, self._cursor(self.active_hand) + duration),
            pending_join=None,
        )

    def _apply_hold(
        self,
        token: HoldToken,
        *,
        duration_vocabulary: DurationVocabulary,
    ) -> GenerationConstraintState:
        self._raise_if_complete_bar_count(token_name="HoldToken")
        duration = self._checked_duration(token.duration_id, duration_vocabulary=duration_vocabulary)
        if duration > self._remaining_duration(self.active_hand):
            raise GenerationConstraintError("hold duration exceeds remaining active-hand measure time")

        cursor = self._cursor(self.active_hand)
        if self._last_attack_end(self.active_hand) != cursor:
            raise GenerationConstraintError("hold token needs a contiguous previous same-hand note or chord")

        cursor_after = cursor + duration
        state = self.with_cursor(self.active_hand, cursor_after)
        state = state.with_last_attack_end(self.active_hand, cursor_after)
        return replace(state, pending_join=None)

    def _apply_join(self) -> GenerationConstraintState:
        if self.pending_join is None or self.pending_join.hand != self.active_hand:
            raise GenerationConstraintError("join token needs a previous same-hand chord note candidate")

        pending = self.pending_join
        if not self._can_add_note_to_onset(pending.target):
            raise GenerationConstraintError("maximum notes per same-hand onset exceeded")

        joined_cursor = max(pending.cursor_before_note, pending.target.start + pending.duration)
        state = self.with_cursor(self.active_hand, joined_cursor)
        state = state.with_last_attack_end(self.active_hand, pending.target.start + pending.duration)
        state = state.with_last_onset(
            self.active_hand,
            replace(
                pending.target,
                note_count=pending.target.note_count + 1,
                midi_pitches=_append_optional_pitch(pending.target.midi_pitches, pending.midi_pitch),
            ),
        )

        return replace(
            state,
            pending_join=None,
            must_join=False,
        )

    def _apply_bar(self) -> GenerationConstraintState:
        self._raise_if_complete_bar_count(token_name="BarToken")
        next_bar_start = (self.bar_index + 1) * self.constraints.measure_duration
        if self.right_cursor != next_bar_start or self.left_cursor != next_bar_start:
            raise GenerationConstraintError("bar token requires both hand cursors to fill the measure")

        return replace(self, bar_index=self.bar_index + 1, pending_join=None)

    def _apply_end(self) -> GenerationConstraintState:
        if self.bar_index != self.constraints.bar_count:
            raise GenerationConstraintError("end token requires the requested number of complete bars")

        return replace(self, ended=True, pending_join=None)

    def _checked_duration(self, duration_id: int, *, duration_vocabulary: DurationVocabulary) -> Fraction:
        duration = duration_vocabulary.id_to_fraction(duration_id)
        if self.constraints.minimum_duration is not None and duration < self.constraints.minimum_duration:
            raise GenerationConstraintError("duration is shorter than the requested minimum")

        if not self.constraints.allow_dotted_durations and is_dotted_duration(duration):
            raise GenerationConstraintError("dotted durations are disabled")

        return duration

    def _note_midi_pitch(self, token: NoteToken) -> int | None:
        if self.constraints.maximum_pitch_gap_semitones is None:
            return None

        if self.constraints.key_root is None or self.constraints.scale_type is None:
            raise GenerationConstraintError("maximum pitch gap requires key_root and scale_type constraints")

        return note_token_to_midi_pitch(
            token,
            key_root=self.constraints.key_root,
            scale_type=self.constraints.scale_type,
            hand=self.active_hand,
        )

    def _exceeds_pitch_gap(self, midi_pitch: int | None) -> bool:
        maximum = self.constraints.maximum_pitch_gap_semitones
        previous_onset = self._last_onset(self.active_hand)
        if maximum is None or midi_pitch is None or previous_onset is None or not previous_onset.midi_pitches:
            return False

        return min(abs(midi_pitch - previous_pitch) for previous_pitch in previous_onset.midi_pitches) > maximum

    def _exceeds_static_hand_span(self, static_position: int) -> bool:
        maximum = self.constraints.maximum_static_hand_span_degrees
        if maximum is None:
            return False

        positions = (*self._static_positions(self.active_hand), static_position)
        return _static_span(positions) > maximum

    def _remaining_duration(self, hand: Hand) -> Fraction:
        return (self.bar_index + 1) * self.constraints.measure_duration - self._cursor(hand)

    def _join_target(self, duration: Fraction) -> _OnsetState | None:
        target = self._last_onset(self.active_hand)
        cursor = self._cursor(self.active_hand)
        if target is None:
            return None

        if target.duration != duration:
            return None

        if target.start + target.duration != cursor:
            return None

        if not self._can_add_note_to_onset(target):
            return None

        return target

    def _can_add_note_to_onset(self, onset: _OnsetState) -> bool:
        maximum = self.constraints.max_notes_per_onset_per_hand
        return maximum is None or onset.note_count < maximum

    def _raise_if_complete_bar_count(self, *, token_name: str) -> None:
        if self.bar_index >= self.constraints.bar_count:
            raise GenerationConstraintError(f"{token_name} cannot be emitted after requested bars are complete")

    def _cursor(self, hand: Hand) -> Fraction:
        return self.right_cursor if hand == Hand.RIGHT else self.left_cursor

    def with_cursor(self, hand: Hand, cursor: Fraction) -> GenerationConstraintState:
        if hand == Hand.RIGHT:
            return replace(self, right_cursor=cursor)

        return replace(self, left_cursor=cursor)

    def _last_attack_end(self, hand: Hand) -> Fraction | None:
        return self.right_last_attack_end if hand == Hand.RIGHT else self.left_last_attack_end

    def with_last_attack_end(self, hand: Hand, attack_end: Fraction) -> GenerationConstraintState:
        if hand == Hand.RIGHT:
            return replace(self, right_last_attack_end=attack_end)

        return replace(self, left_last_attack_end=attack_end)

    def _last_onset(self, hand: Hand) -> _OnsetState | None:
        return self.right_last_onset if hand == Hand.RIGHT else self.left_last_onset

    def with_last_onset(self, hand: Hand, onset: _OnsetState) -> GenerationConstraintState:
        if hand == Hand.RIGHT:
            return replace(self, right_last_onset=onset)

        return replace(self, left_last_onset=onset)

    def _static_positions(self, hand: Hand) -> tuple[int, ...]:
        return self.right_static_positions if hand == Hand.RIGHT else self.left_static_positions

    def with_static_position(self, hand: Hand, position: int) -> GenerationConstraintState:
        if hand == Hand.RIGHT:
            return replace(self, right_static_positions=(*self.right_static_positions, position))

        return replace(self, left_static_positions=(*self.left_static_positions, position))


def state_from_tokens(
    tokens: Sequence[Token],
    *,
    constraints: GenerationConstraints,
    duration_vocabulary: DurationVocabulary,
) -> GenerationConstraintState:
    state = GenerationConstraintState(constraints=constraints)
    for token in tokens:
        state = state.apply(
            token,
            duration_vocabulary=duration_vocabulary,
        )

    return state


def state_from_token_ids(
    token_ids: Sequence[int],
    *,
    constraints: GenerationConstraints,
    token_vocabulary: TokenVocabulary,
    duration_vocabulary: DurationVocabulary,
) -> GenerationConstraintState:
    return state_from_tokens(
        token_vocabulary.decode(list(token_ids)),
        constraints=constraints,
        duration_vocabulary=duration_vocabulary,
    )


def allowed_next_token_ids(
    prefix_token_ids: Sequence[int],
    *,
    constraints: GenerationConstraints,
    token_vocabulary: TokenVocabulary,
    duration_vocabulary: DurationVocabulary,
) -> frozenset[int]:
    state = state_from_token_ids(
        prefix_token_ids,
        constraints=constraints,
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
    )

    return state.allowed_token_ids(
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
    )


def mask_disallowed_logits(
    logits: Tensor,
    *,
    allowed_token_ids: Collection[int],
    mask_value: float = float("-inf"),
) -> Tensor:
    if not allowed_token_ids:
        raise GenerationConstraintError("cannot mask logits without any allowed token IDs")

    masked = torch.full_like(logits, mask_value)
    allowed_indices = torch.tensor(sorted(allowed_token_ids), dtype=torch.long, device=logits.device)
    masked[..., allowed_indices] = logits[..., allowed_indices]
    return masked


def _optional_pitch_tuple(midi_pitch: int | None) -> tuple[int, ...]:
    if midi_pitch is None:
        return ()

    return (midi_pitch,)


def _append_optional_pitch(midi_pitches: tuple[int, ...], midi_pitch: int | None) -> tuple[int, ...]:
    if midi_pitch is None:
        return midi_pitches

    return (*midi_pitches, midi_pitch)


def _static_span(positions: tuple[int, ...]) -> int:
    if len(positions) < 2:
        return 0

    return max(positions) - min(positions) + 1
