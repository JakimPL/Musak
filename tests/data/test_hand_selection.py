from dataclasses import dataclass
from typing import Callable

import pytest
from music21 import chord, instrument
from music21.note import Note, Rest
from music21.stream.base import Measure, Part, Score

from musak_model.data.hand_selection import (
    AmbiguousHandAssignmentError,
    EmptyPitchedPartError,
    InvalidPianoPartCountError,
    NonPianoInstrumentError,
    PianoHandSelectionError,
    select_piano_hand_parts,
)


@dataclass(frozen=True)
class PartCountCase:
    label: str
    parts: tuple[Part, ...]
    expected_message: str


@dataclass(frozen=True)
class InstrumentCase:
    label: str
    left_instrument: instrument.Instrument | None
    right_instrument: instrument.Instrument | None
    expected_error: type[PianoHandSelectionError] | None = None


@dataclass(frozen=True)
class AssignmentCase:
    label: str
    first_part: Part
    second_part: Part
    expected_right_index: int


@dataclass(frozen=True)
class RejectionCase:
    label: str
    score_factory: Callable[[], Score]
    expected_error: type[PianoHandSelectionError]
    expected_message: str


def _part_with_notes(
    *pitch_names: str,
    part_instrument: instrument.Instrument | None = None,
) -> Part:
    part = Part()
    if part_instrument is not None:
        part.insert(0, part_instrument)
    measure = Measure(number=1)
    for offset, pitch_name in enumerate(pitch_names):
        measure.insert(offset, Note(pitch_name, quarterLength=1))
    part.append(measure)
    return part


def _part_with_chord(
    pitch_names: list[str],
    *,
    part_instrument: instrument.Instrument | None = None,
) -> Part:
    part = Part()
    if part_instrument is not None:
        part.insert(0, part_instrument)
    measure = Measure(number=1)
    measure.insert(0, chord.Chord(pitch_names, quarterLength=1))
    part.append(measure)
    return part


def _instrument_with_midi_program(midi_program: int) -> instrument.Instrument:
    part_instrument = instrument.Instrument()
    part_instrument.midiProgram = midi_program
    return part_instrument


def _rest_only_part() -> Part:
    part = Part()
    measure = Measure(number=1)
    measure.insert(0, Rest(quarterLength=1))
    part.append(measure)
    return part


def _score(*parts: Part) -> Score:
    score = Score()
    for part in parts:
        score.insert(0, part)
    return score


class TestPartCountPolicy:
    CASES = (
        PartCountCase(
            label="one_part",
            parts=(_part_with_notes("C4"),),
            expected_message="exactly 2",
        ),
        PartCountCase(
            label="three_parts",
            parts=(_part_with_notes("C3"), _part_with_notes("C4"), _part_with_notes("C5")),
            expected_message="exactly 2",
        ),
    )

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
    def test_rejects_part_counts_other_than_two(self, case: PartCountCase) -> None:
        with pytest.raises(InvalidPianoPartCountError, match=case.expected_message):
            select_piano_hand_parts(_score(*case.parts))


class TestInstrumentPolicy:
    CASES = (
        InstrumentCase(
            label="missing_instruments_allowed",
            left_instrument=None,
            right_instrument=None,
        ),
        InstrumentCase(
            label="explicit_pianos_allowed",
            left_instrument=instrument.Piano(),
            right_instrument=instrument.Piano(),
        ),
        InstrumentCase(
            label="midi_program_zero_allowed",
            left_instrument=_instrument_with_midi_program(0),
            right_instrument=instrument.Piano(),
        ),
        InstrumentCase(
            label="explicit_violin_rejected",
            left_instrument=instrument.Violin(),
            right_instrument=instrument.Piano(),
            expected_error=NonPianoInstrumentError,
        ),
        InstrumentCase(
            label="explicit_keyboard_non_piano_rejected",
            left_instrument=instrument.Harpsichord(),
            right_instrument=instrument.Piano(),
            expected_error=NonPianoInstrumentError,
        ),
    )

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
    def test_instrument_policy(self, case: InstrumentCase) -> None:
        score = _score(
            _part_with_notes("C3", part_instrument=case.left_instrument),
            _part_with_notes("C5", part_instrument=case.right_instrument),
        )

        if case.expected_error is not None:
            with pytest.raises(case.expected_error, match="non-piano"):
                select_piano_hand_parts(score)
            return

        hand_parts = select_piano_hand_parts(score)
        assert hand_parts.right is score.parts[1]
        assert hand_parts.left is score.parts[0]


class TestPitchCenterAssignment:
    CASES = (
        AssignmentCase(
            label="score_order_left_then_right",
            first_part=_part_with_notes("C3"),
            second_part=_part_with_notes("C5"),
            expected_right_index=1,
        ),
        AssignmentCase(
            label="score_order_right_then_left",
            first_part=_part_with_notes("C5"),
            second_part=_part_with_notes("C3"),
            expected_right_index=0,
        ),
        AssignmentCase(
            label="median_ignores_high_outlier",
            first_part=_part_with_notes("C3", "D3", "C7"),
            second_part=_part_with_notes("C4", "D4", "E4"),
            expected_right_index=1,
        ),
        AssignmentCase(
            label="chord_pitches_count_toward_center",
            first_part=_part_with_chord(["C3", "E3", "G3"]),
            second_part=_part_with_chord(["C5", "E5", "G5"]),
            expected_right_index=1,
        ),
    )

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
    def test_assigns_right_hand_to_higher_pitch_center(self, case: AssignmentCase) -> None:
        score = _score(case.first_part, case.second_part)

        hand_parts = select_piano_hand_parts(score)

        assert hand_parts.right is score.parts[case.expected_right_index]
        assert hand_parts.left is score.parts[1 - case.expected_right_index]


class TestPitchCenterRejection:
    CASES = (
        RejectionCase(
            label="empty_part",
            score_factory=lambda: _score(Part(), _part_with_notes("C4")),
            expected_error=EmptyPitchedPartError,
            expected_message="no pitched events",
        ),
        RejectionCase(
            label="rest_only_part",
            score_factory=lambda: _score(_rest_only_part(), _part_with_notes("C4")),
            expected_error=EmptyPitchedPartError,
            expected_message="no pitched events",
        ),
        RejectionCase(
            label="identical_pitch_centers",
            score_factory=lambda: _score(_part_with_notes("C4"), _part_with_notes("C4")),
            expected_error=AmbiguousHandAssignmentError,
            expected_message="identical pitch centers",
        ),
        RejectionCase(
            label="rounded_pitch_centers_are_identical",
            score_factory=lambda: _score(_part_with_notes("C4", "C#4"), _part_with_notes("C4")),
            expected_error=AmbiguousHandAssignmentError,
            expected_message="identical pitch centers",
        ),
    )

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
    def test_rejects_unusable_pitch_centers(self, case: RejectionCase) -> None:
        with pytest.raises(case.expected_error, match=case.expected_message):
            select_piano_hand_parts(case.score_factory())
