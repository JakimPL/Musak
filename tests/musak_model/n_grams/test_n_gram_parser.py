from fractions import Fraction

import pytest

from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
)


def test_rest_splits_hand_onset_runs(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    runs = extract_hand_onset_runs(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, duration_id=quarter_id),
            RestToken(duration_id=quarter_id),
            _note(2, duration_id=quarter_id),
        ],
        duration_vocabulary=duration_vocabulary,
        time_numerator=4,
        time_denominator=4,
    )

    assert [[onset.notes[0].degree for onset in run.onsets] for run in runs[Hand.RIGHT]] == [[1], [2]]
    assert runs[Hand.LEFT] == ()


def test_barline_splits_runs_and_resets_hand_time(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    runs = extract_hand_onset_runs(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, duration_id=quarter_id),
            BarToken(),
            _note(2, duration_id=quarter_id),
        ],
        duration_vocabulary=duration_vocabulary,
        time_numerator=4,
        time_denominator=4,
    )

    assert len(runs[Hand.RIGHT]) == 2
    assert runs[Hand.RIGHT][0].onsets[0].start == Fraction(0)
    assert runs[Hand.RIGHT][1].onsets[0].start == Fraction(1)


def test_join_with_previous_groups_same_hand_chord(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    runs = extract_hand_onset_runs(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, duration_id=quarter_id),
            _note(3, duration_id=quarter_id),
            JoinWithPreviousToken(),
        ],
        duration_vocabulary=duration_vocabulary,
        time_numerator=4,
        time_denominator=4,
    )

    onset = runs[Hand.RIGHT][0].onsets[0]
    assert [note.degree for note in onset.notes] == [1, 3]
    assert onset.start == Fraction(0)
    assert onset.duration == Fraction(1, 4)


def test_join_with_previous_does_not_merge_hands(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    runs = extract_hand_onset_runs(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, duration_id=quarter_id),
            HandToken(hand=Hand.LEFT),
            _note(5, duration_id=quarter_id),
            JoinWithPreviousToken(),
        ],
        duration_vocabulary=duration_vocabulary,
        time_numerator=4,
        time_denominator=4,
    )

    assert [note.degree for note in runs[Hand.RIGHT][0].onsets[0].notes] == [1]
    assert [note.degree for note in runs[Hand.LEFT][0].onsets[0].notes] == [5]
    assert runs[Hand.RIGHT][0].onsets[0].start == runs[Hand.LEFT][0].onsets[0].start


def test_hold_extends_previous_same_hand_onset(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    runs = extract_hand_onset_runs(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, duration_id=quarter_id),
            HoldToken(duration_id=quarter_id),
        ],
        duration_vocabulary=duration_vocabulary,
        time_numerator=4,
        time_denominator=4,
    )

    assert runs[Hand.RIGHT][0].onsets[0].duration == Fraction(1, 2)


def test_hold_without_previous_attack_fails(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))

    with pytest.raises(ValueError, match="hold token has no previous right hand attack"):
        extract_hand_onset_runs(
            [HandToken(hand=Hand.RIGHT), HoldToken(duration_id=quarter_id)],
            duration_vocabulary=duration_vocabulary,
            time_numerator=4,
            time_denominator=4,
        )


def _note(degree: int, *, duration_id: int) -> NoteToken:
    return NoteToken(
        degree=degree,
        accidental=0,
        octave_offset=0,
        duration_id=duration_id,
    )
