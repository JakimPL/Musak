from fractions import Fraction

from musak_model.n_grams.figure.parser import HandOnsetRun, PitchedOnset
from musak_model.n_grams.figure.signature import (
    iter_figure_occurrences_from_run,
    iter_figure_signatures_from_run,
)
from musak_model.tokens.schema import Hand, NoteToken

type NoteSpec = tuple[int, int, int]


def _onset(note_specs: list[NoteSpec], *, start: Fraction, duration: Fraction) -> PitchedOnset:
    notes = tuple(
        NoteToken(degree=degree, accidental=accidental, octave_offset=octave_offset, duration_id=0)
        for degree, accidental, octave_offset in note_specs
    )
    return PitchedOnset(notes=notes, start=start, duration=duration)


def _run(onsets: list[PitchedOnset]) -> HandOnsetRun:
    return HandOnsetRun(hand=Hand.RIGHT, onsets=tuple(onsets))


def test_occurrence_exposes_anchor_base_duration_and_start() -> None:
    run = _run(
        [
            _onset([(3, 0, 1)], start=Fraction(1, 2), duration=Fraction(1, 4)),
            _onset([(5, 0, 1)], start=Fraction(3, 4), duration=Fraction(1, 8)),
        ]
    )

    occurrences = list(iter_figure_occurrences_from_run(run, min_n=2, max_n=2, scale_size=7))

    assert len(occurrences) == 1
    occurrence = occurrences[0]
    assert occurrence.figure_length == 2
    assert (occurrence.anchor_degree, occurrence.anchor_accidental, occurrence.anchor_octave) == (3, 0, 1)
    assert occurrence.base_duration == Fraction(1, 8)
    assert occurrence.start == Fraction(1, 2)


def test_occurrence_anchor_handles_negative_octave_and_chord_accidental() -> None:
    run = _run(
        [
            _onset([(7, 0, -1)], start=Fraction(0), duration=Fraction(1, 4)),
            _onset([(2, -1, 0), (4, 0, 0)], start=Fraction(1, 4), duration=Fraction(1, 4)),
        ]
    )

    octave_down = next(iter(iter_figure_occurrences_from_run(run, min_n=1, max_n=1, scale_size=7)))

    assert (octave_down.anchor_degree, octave_down.anchor_octave) == (7, -1)


def test_occurrence_anchor_uses_lowest_note_accidental() -> None:
    run = _run([_onset([(3, -1, 0), (5, 0, 0)], start=Fraction(0), duration=Fraction(1, 4))])

    occurrence = next(iter(iter_figure_occurrences_from_run(run, min_n=1, max_n=1, scale_size=7)))

    assert (occurrence.anchor_degree, occurrence.anchor_accidental, occurrence.anchor_octave) == (3, -1, 0)


def test_signature_iterator_matches_occurrence_signatures() -> None:
    run = _run(
        [
            _onset([(1, 0, 0)], start=Fraction(0), duration=Fraction(1, 4)),
            _onset([(2, 0, 0)], start=Fraction(1, 4), duration=Fraction(1, 4)),
            _onset([(3, 0, 0)], start=Fraction(1, 2), duration=Fraction(1, 4)),
        ]
    )

    signatures = list(iter_figure_signatures_from_run(run, min_n=2, max_n=3, scale_size=7))
    occurrences = list(iter_figure_occurrences_from_run(run, min_n=2, max_n=3, scale_size=7))

    assert signatures == [(occurrence.figure_length, occurrence.signature) for occurrence in occurrences]
