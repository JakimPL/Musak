from fractions import Fraction

import pytest
import torch

from musak_model.generation.constraints import (
    GenerationConstraintError,
    GenerationConstraints,
    allowed_next_token_ids,
    mask_disallowed_logits,
    state_from_tokens,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    ScaleType,
    StartToken,
)
from musak_model.tokens.vocabulary import TokenVocabulary


def _constraints(**overrides: object) -> GenerationConstraints:
    values = {
        "time_numerator": 4,
        "time_denominator": 4,
        "bar_count": 1,
    }
    values.update(overrides)
    return GenerationConstraints(**values)


def _note(duration_id: int) -> NoteToken:
    return NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=duration_id)


def _ids(tokens: list[object], *, token_vocabulary: TokenVocabulary) -> list[int]:
    return [token_vocabulary.token_to_id(token) for token in tokens]


class TestAllowedNextTokenIds:
    def test_disallows_notes_and_rests_that_exceed_remaining_measure_time(
        self,
        duration_vocabulary: DurationVocabulary,
        token_vocabulary: TokenVocabulary,
    ) -> None:
        half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))
        whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
        prefix = _ids([_note(half_id)], token_vocabulary=token_vocabulary)

        allowed = allowed_next_token_ids(
            prefix,
            constraints=_constraints(),
            token_vocabulary=token_vocabulary,
            duration_vocabulary=duration_vocabulary,
        )

        assert token_vocabulary.token_to_id(_note(half_id)) in allowed
        assert token_vocabulary.token_to_id(_note(whole_id)) not in allowed
        assert token_vocabulary.token_to_id(RestToken(duration_id=whole_id)) not in allowed

    def test_allows_bar_only_when_both_hands_fill_measure(
        self,
        duration_vocabulary: DurationVocabulary,
        token_vocabulary: TokenVocabulary,
    ) -> None:
        whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
        prefix = _ids(
            [
                HandToken(hand=Hand.RIGHT),
                _note(whole_id),
                HandToken(hand=Hand.LEFT),
                RestToken(duration_id=whole_id),
            ],
            token_vocabulary=token_vocabulary,
        )

        allowed = allowed_next_token_ids(
            prefix,
            constraints=_constraints(),
            token_vocabulary=token_vocabulary,
            duration_vocabulary=duration_vocabulary,
        )

        assert token_vocabulary.bar_token_id in allowed
        assert token_vocabulary.end_token_id not in allowed

    def test_allows_end_only_after_requested_complete_bars(
        self,
        duration_vocabulary: DurationVocabulary,
        token_vocabulary: TokenVocabulary,
    ) -> None:
        whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
        prefix = _ids(
            [
                HandToken(hand=Hand.RIGHT),
                _note(whole_id),
                HandToken(hand=Hand.LEFT),
                RestToken(duration_id=whole_id),
                BarToken(),
            ],
            token_vocabulary=token_vocabulary,
        )

        allowed = allowed_next_token_ids(
            prefix,
            constraints=_constraints(),
            token_vocabulary=token_vocabulary,
            duration_vocabulary=duration_vocabulary,
        )

        assert allowed == frozenset({token_vocabulary.end_token_id})

    def test_allows_hold_across_bar_only_when_same_hand_attack_is_contiguous(
        self,
        duration_vocabulary: DurationVocabulary,
        token_vocabulary: TokenVocabulary,
    ) -> None:
        whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
        half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))
        prefix = _ids(
            [
                HandToken(hand=Hand.RIGHT),
                _note(whole_id),
                HandToken(hand=Hand.LEFT),
                RestToken(duration_id=whole_id),
                BarToken(),
                HandToken(hand=Hand.RIGHT),
            ],
            token_vocabulary=token_vocabulary,
        )

        allowed = allowed_next_token_ids(
            prefix,
            constraints=_constraints(bar_count=2),
            token_vocabulary=token_vocabulary,
            duration_vocabulary=duration_vocabulary,
        )

        assert token_vocabulary.token_to_id(HoldToken(duration_id=half_id)) in allowed

    def test_disallows_hold_after_same_hand_gap(
        self,
        duration_vocabulary: DurationVocabulary,
        token_vocabulary: TokenVocabulary,
    ) -> None:
        quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
        prefix = _ids(
            [
                HandToken(hand=Hand.RIGHT),
                _note(quarter_id),
                RestToken(duration_id=quarter_id),
            ],
            token_vocabulary=token_vocabulary,
        )

        allowed = allowed_next_token_ids(
            prefix,
            constraints=_constraints(),
            token_vocabulary=token_vocabulary,
            duration_vocabulary=duration_vocabulary,
        )

        assert token_vocabulary.token_to_id(HoldToken(duration_id=quarter_id)) not in allowed

    def test_bar_ending_chord_note_must_join_before_any_other_token(
        self,
        duration_vocabulary: DurationVocabulary,
        token_vocabulary: TokenVocabulary,
    ) -> None:
        whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
        prefix = _ids(
            [
                HandToken(hand=Hand.RIGHT),
                _note(whole_id),
                _note(whole_id),
            ],
            token_vocabulary=token_vocabulary,
        )

        allowed = allowed_next_token_ids(
            prefix,
            constraints=_constraints(),
            token_vocabulary=token_vocabulary,
            duration_vocabulary=duration_vocabulary,
        )

        assert allowed == frozenset({token_vocabulary.join_with_previous_token_id})

    def test_respects_minimum_duration(
        self,
        duration_vocabulary: DurationVocabulary,
        token_vocabulary: TokenVocabulary,
    ) -> None:
        quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
        eighth_id = duration_vocabulary.fraction_to_id(Fraction(1, 8))

        allowed = allowed_next_token_ids(
            [],
            constraints=_constraints(minimum_duration=Fraction(1, 4)),
            token_vocabulary=token_vocabulary,
            duration_vocabulary=duration_vocabulary,
        )

        assert token_vocabulary.token_to_id(_note(quarter_id)) in allowed
        assert token_vocabulary.token_to_id(_note(eighth_id)) not in allowed

    def test_can_disable_dotted_durations(
        self,
        duration_vocabulary: DurationVocabulary,
        token_vocabulary: TokenVocabulary,
    ) -> None:
        dotted_quarter_id = duration_vocabulary.fraction_to_id(Fraction(3, 8))

        allowed = allowed_next_token_ids(
            [],
            constraints=_constraints(allow_dotted_durations=False),
            token_vocabulary=token_vocabulary,
            duration_vocabulary=duration_vocabulary,
        )

        assert token_vocabulary.token_to_id(_note(dotted_quarter_id)) not in allowed

    def test_disallows_melodic_gap_above_maximum(
        self,
        duration_vocabulary: DurationVocabulary,
        token_vocabulary: TokenVocabulary,
    ) -> None:
        quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
        prefix = _ids(
            [
                _note(quarter_id),
                RestToken(duration_id=quarter_id),
            ],
            token_vocabulary=token_vocabulary,
        )
        nearby_note = NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=quarter_id)
        distant_note = NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=quarter_id)

        allowed = allowed_next_token_ids(
            prefix,
            constraints=_constraints(
                maximum_pitch_gap_semitones=2,
                key_root=0,
                scale_type=ScaleType.MAJOR,
            ),
            token_vocabulary=token_vocabulary,
            duration_vocabulary=duration_vocabulary,
        )

        assert token_vocabulary.token_to_id(nearby_note) in allowed
        assert token_vocabulary.token_to_id(distant_note) not in allowed

    def test_large_chord_interval_is_allowed_only_when_joined(
        self,
        duration_vocabulary: DurationVocabulary,
        token_vocabulary: TokenVocabulary,
    ) -> None:
        quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
        distant_note = NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=quarter_id)
        prefix = _ids(
            [
                _note(quarter_id),
                distant_note,
            ],
            token_vocabulary=token_vocabulary,
        )

        allowed = allowed_next_token_ids(
            prefix,
            constraints=_constraints(
                maximum_pitch_gap_semitones=2,
                key_root=0,
                scale_type=ScaleType.MAJOR,
            ),
            token_vocabulary=token_vocabulary,
            duration_vocabulary=duration_vocabulary,
        )

        assert allowed == frozenset({token_vocabulary.join_with_previous_token_id})

    def test_maximum_pitch_gap_requires_key_and_scale(
        self,
        duration_vocabulary: DurationVocabulary,
        token_vocabulary: TokenVocabulary,
    ) -> None:
        quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))

        with pytest.raises(GenerationConstraintError, match="requires key_root and scale_type"):
            allowed_next_token_ids(
                _ids([_note(quarter_id)], token_vocabulary=token_vocabulary),
                constraints=_constraints(maximum_pitch_gap_semitones=2),
                token_vocabulary=token_vocabulary,
                duration_vocabulary=duration_vocabulary,
            )


class TestGenerationConstraintState:
    def test_join_restores_cursor_after_bar_ending_chord_note(
        self,
        duration_vocabulary: DurationVocabulary,
    ) -> None:
        whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))

        state = state_from_tokens(
            [
                HandToken(hand=Hand.RIGHT),
                _note(whole_id),
                _note(whole_id),
                JoinWithPreviousToken(),
            ],
            constraints=_constraints(),
            duration_vocabulary=duration_vocabulary,
        )

        assert state.right_cursor == Fraction(1, 1)

    def test_start_token_is_accepted_in_prefix_but_never_allowed_as_next_token(
        self,
        duration_vocabulary: DurationVocabulary,
        token_vocabulary: TokenVocabulary,
    ) -> None:
        allowed = allowed_next_token_ids(
            _ids([StartToken()], token_vocabulary=token_vocabulary),
            constraints=_constraints(),
            token_vocabulary=token_vocabulary,
            duration_vocabulary=duration_vocabulary,
        )

        assert token_vocabulary.start_token_id not in allowed

    def test_invalid_prefix_raises_constraint_error(self, duration_vocabulary: DurationVocabulary) -> None:
        whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))

        with pytest.raises(GenerationConstraintError, match="both hand cursors"):
            state_from_tokens(
                [HandToken(hand=Hand.RIGHT), _note(whole_id), BarToken()],
                constraints=_constraints(),
                duration_vocabulary=duration_vocabulary,
            )


class TestMaskDisallowedLogits:
    def test_keeps_allowed_logits_and_masks_the_rest(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        masked = mask_disallowed_logits(logits, allowed_token_ids={0, 2})

        assert masked.tolist() == [[1.0, float("-inf"), 3.0]]

    def test_rejects_empty_allowed_token_ids(self) -> None:
        with pytest.raises(GenerationConstraintError, match="without any allowed"):
            mask_disallowed_logits(torch.tensor([1.0, 2.0]), allowed_token_ids=set())
