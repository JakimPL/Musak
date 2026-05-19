from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from typing import Final

import torch
from torch import Tensor

from musak_model.generation.constraints import (
    MAX_NOTES_PER_HAND,
    GenerationConstraintError,
    GenerationConstraints,
    GenerationConstraintState,
)
from musak_model.tokens.pitch import note_token_to_midi_pitch
from musak_model.tokens.schema import Hand, HoldToken, NoteToken, RestToken, ScaleType, StartToken
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.dataset import TrainingBatch

MAX_ONSET_SPAN_SEMITONES: Final[int] = 12
_INVALID_MIDI_PITCH: Final[int] = -1_000_000


@dataclass(frozen=True)
class ValidityPenaltyMasks:
    invalid_token_mask: Tensor
    invalid_target_mask: Tensor


class TrainingValidityMaskBuilder:
    def __init__(self, token_vocabulary: TokenVocabulary) -> None:
        self._token_vocabulary = token_vocabulary
        self._duration_tick_denominator = _duration_tick_denominator(token_vocabulary)
        self._metadata = _TokenValidityMetadata.build(
            token_vocabulary,
            duration_tick_denominator=self._duration_tick_denominator,
        )

    def masks_for_batch(self, batch: TrainingBatch, *, device: torch.device) -> ValidityPenaltyMasks:
        batch_size, sequence_length = batch.input_token_ids.shape
        invalid_token_mask = torch.zeros(
            (batch_size, sequence_length, self._token_vocabulary.vocabulary_size),
            dtype=torch.bool,
        )
        invalid_target_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool)
        input_token_ids = batch.input_token_ids.detach().cpu()
        target_token_ids = batch.target_token_ids.detach().cpu()
        token_padding_mask = batch.token_padding_mask.detach().cpu()

        for row_index in range(batch_size):
            constraints = GenerationConstraints(
                time_numerator=int(batch.time_numerators[row_index].item()),
                time_denominator=int(batch.time_denominators[row_index].item()),
                bar_count=int(batch.bar_counts[row_index].item()),
                max_notes_per_hand=MAX_NOTES_PER_HAND,
                maximum_onset_span_semitones=MAX_ONSET_SPAN_SEMITONES,
                key_root=int(batch.key_roots[row_index].item()),
                scale_type=tuple(ScaleType)[int(batch.scale_type_ids[row_index].item())],
            )
            state = GenerationConstraintState(constraints=constraints)
            prefix_is_valid = True
            for token_index in range(sequence_length):
                if bool(token_padding_mask[row_index, token_index].item()):
                    break

                if prefix_is_valid:
                    input_token_id = int(input_token_ids[row_index, token_index].item())
                    input_token = self._token_vocabulary.id_to_token(input_token_id)
                    try:
                        state = state.apply(
                            input_token,
                            duration_vocabulary=self._token_vocabulary.duration_vocabulary,
                        )
                    except GenerationConstraintError:
                        prefix_is_valid = False

                if not prefix_is_valid:
                    continue

                row_mask = self._invalid_token_mask(state)
                invalid_token_mask[row_index, token_index] = row_mask
                target_token_id = int(target_token_ids[row_index, token_index].item())
                invalid_target_mask[row_index, token_index] = row_mask[target_token_id]

        return ValidityPenaltyMasks(
            invalid_token_mask=invalid_token_mask.to(device),
            invalid_target_mask=invalid_target_mask.to(device),
        )

    def _invalid_token_mask(self, state: GenerationConstraintState) -> Tensor:
        metadata = self._metadata
        invalid = torch.zeros(self._token_vocabulary.vocabulary_size, dtype=torch.bool)
        invalid |= metadata.is_start

        if state.ended:
            return torch.ones_like(invalid)

        if state.must_join:
            invalid = torch.ones_like(invalid)
            invalid[self._token_vocabulary.join_with_previous_token_id] = False
            return invalid

        if state.bar_index >= state.constraints.bar_count:
            invalid |= ~metadata.is_end
            return invalid

        remaining_ticks = self._fraction_to_ticks(state.remaining_duration(state.active_hand))
        invalid |= (metadata.is_rest | metadata.is_hold) & (metadata.duration_ticks > remaining_ticks)
        invalid |= metadata.is_hold & (state.last_attack_end(state.active_hand) != state.cursor(state.active_hand))
        invalid |= metadata.is_bar & (not self._can_emit_bar(state))
        invalid |= metadata.is_end & (state.bar_index != state.constraints.bar_count)
        invalid |= metadata.is_join & (not self._can_join_pending(state))
        invalid |= self._invalid_note_mask(state, remaining_ticks=remaining_ticks)
        return invalid

    def _invalid_note_mask(self, state: GenerationConstraintState, *, remaining_ticks: int) -> Tensor:
        metadata = self._metadata
        duration_can_join = self._joinable_duration_mask(state)
        exceeds_remaining = metadata.duration_ticks > remaining_ticks
        forced_join = duration_can_join & exceeds_remaining
        invalid = metadata.is_note & exceeds_remaining & ~duration_can_join
        if not bool(duration_can_join.any().item()):
            return invalid

        midi_pitches = metadata.midi_pitches[
            int(tuple(ScaleType).index(state.constraints.scale_type or ScaleType.MAJOR)),
            int(state.constraints.key_root or 0),
            _hand_index(state.active_hand),
        ]
        onset = state.last_onset(state.active_hand)
        if onset is None:
            return invalid

        for previous_pitch in onset.midi_pitches:
            invalid |= forced_join & (midi_pitches == previous_pitch)

        if onset.midi_pitches:
            low = min(onset.midi_pitches)
            high = max(onset.midi_pitches)
            joined_low = torch.minimum(midi_pitches, torch.full_like(midi_pitches, low))
            joined_high = torch.maximum(midi_pitches, torch.full_like(midi_pitches, high))
            invalid |= forced_join & ((joined_high - joined_low) > MAX_ONSET_SPAN_SEMITONES)

        return invalid

    def _joinable_duration_mask(self, state: GenerationConstraintState) -> Tensor:
        target = state.last_onset(state.active_hand)
        if target is None:
            return torch.zeros(self._token_vocabulary.vocabulary_size, dtype=torch.bool)

        if target.start + target.duration != state.cursor(state.active_hand):
            return torch.zeros(self._token_vocabulary.vocabulary_size, dtype=torch.bool)

        if target.note_count >= MAX_NOTES_PER_HAND:
            return torch.zeros(self._token_vocabulary.vocabulary_size, dtype=torch.bool)

        return self._metadata.is_note & (self._metadata.duration_ticks == self._fraction_to_ticks(target.duration))

    def _can_join_pending(self, state: GenerationConstraintState) -> bool:
        pending = state.pending_join
        if pending is None or pending.hand != state.active_hand:
            return False

        if pending.target.note_count >= MAX_NOTES_PER_HAND:
            return False

        if pending.midi_pitch is None:
            return True

        if pending.midi_pitch in pending.target.midi_pitches:
            return False

        pitches = (*pending.target.midi_pitches, pending.midi_pitch)
        return max(pitches) - min(pitches) <= MAX_ONSET_SPAN_SEMITONES

    def _can_emit_bar(self, state: GenerationConstraintState) -> bool:
        next_bar_start = (state.bar_index + 1) * state.constraints.measure_duration
        return state.right_cursor == next_bar_start and state.left_cursor == next_bar_start

    def _fraction_to_ticks(self, value: Fraction) -> int:
        ticks = value * self._duration_tick_denominator
        if ticks.denominator != 1:
            raise ValueError(f"duration {value} cannot be represented as integer ticks")

        return ticks.numerator


@dataclass(frozen=True)
class _TokenValidityMetadata:
    is_note: Tensor
    is_rest: Tensor
    is_hold: Tensor
    is_bar: Tensor
    is_end: Tensor
    is_join: Tensor
    is_start: Tensor
    duration_ticks: Tensor
    midi_pitches: Tensor

    @classmethod
    def build(cls, token_vocabulary: TokenVocabulary, *, duration_tick_denominator: int) -> _TokenValidityMetadata:
        vocabulary_size = token_vocabulary.vocabulary_size
        is_note = torch.zeros(vocabulary_size, dtype=torch.bool)
        is_rest = torch.zeros(vocabulary_size, dtype=torch.bool)
        is_hold = torch.zeros(vocabulary_size, dtype=torch.bool)
        is_bar = torch.zeros(vocabulary_size, dtype=torch.bool)
        is_end = torch.zeros(vocabulary_size, dtype=torch.bool)
        is_join = torch.zeros(vocabulary_size, dtype=torch.bool)
        is_start = torch.zeros(vocabulary_size, dtype=torch.bool)
        duration_ticks = torch.zeros(vocabulary_size, dtype=torch.long)
        midi_pitches = torch.full(
            (len(ScaleType), 12, len(Hand), vocabulary_size),
            _INVALID_MIDI_PITCH,
            dtype=torch.long,
        )

        for token_id in range(vocabulary_size):
            token = token_vocabulary.id_to_token(token_id)
            match token:
                case NoteToken():
                    is_note[token_id] = True
                    duration_ticks[token_id] = _duration_ticks(
                        token_vocabulary.duration_vocabulary.id_to_fraction(token.duration_id),
                        denominator=duration_tick_denominator,
                    )
                    for scale_index, scale_type in enumerate(ScaleType):
                        for key_root in range(12):
                            for hand_index, hand in enumerate(Hand):
                                midi_pitches[scale_index, key_root, hand_index, token_id] = note_token_to_midi_pitch(
                                    token,
                                    key_root=key_root,
                                    scale_type=scale_type,
                                    hand=hand,
                                )
                case RestToken():
                    is_rest[token_id] = True
                    duration_ticks[token_id] = _duration_ticks(
                        token_vocabulary.duration_vocabulary.id_to_fraction(token.duration_id),
                        denominator=duration_tick_denominator,
                    )
                case HoldToken():
                    is_hold[token_id] = True
                    duration_ticks[token_id] = _duration_ticks(
                        token_vocabulary.duration_vocabulary.id_to_fraction(token.duration_id),
                        denominator=duration_tick_denominator,
                    )
                case StartToken():
                    is_start[token_id] = True
                case _:
                    pass

        is_bar[token_vocabulary.bar_token_id] = True
        is_end[token_vocabulary.end_token_id] = True
        is_join[token_vocabulary.join_with_previous_token_id] = True
        return cls(
            is_note=is_note,
            is_rest=is_rest,
            is_hold=is_hold,
            is_bar=is_bar,
            is_end=is_end,
            is_join=is_join,
            is_start=is_start,
            duration_ticks=duration_ticks,
            midi_pitches=midi_pitches,
        )


def _duration_tick_denominator(token_vocabulary: TokenVocabulary) -> int:
    denominators = [
        token_vocabulary.duration_vocabulary.id_to_fraction(duration_id).denominator
        for duration_id in range(token_vocabulary.duration_vocabulary.vocabulary_size())
    ]
    return lcm(*denominators)


def _duration_ticks(value: Fraction, *, denominator: int) -> int:
    ticks = value * denominator
    if ticks.denominator != 1:
        raise ValueError(f"duration {value} cannot be represented as integer ticks")

    return ticks.numerator


def _hand_index(hand: Hand) -> int:
    return 0 if hand == Hand.RIGHT else 1
