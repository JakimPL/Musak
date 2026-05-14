from fractions import Fraction

from musak_model.data.cleaning import (
    clean_parsed_score,
    deduplicate_simultaneous_pitches,
    trim_silent_edge_bars,
    truncate_overlapping_events,
)
from musak_model.data.schema import ParsedChord, ParsedNote
from tests.data.fixtures import bar, chord_event, note_event, parsed_score, rest_event


def _note_bar():
    return bar([note_event(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0))])


def _chord_bar():
    return bar([chord_event(midi_pitches=[60, 64], duration=Fraction(1, 4), beat_offset=Fraction(0))])


def _rest_bar():
    return bar([rest_event(duration=Fraction(1, 1), beat_offset=Fraction(0))])


def test_trim_silent_edge_bars_removes_only_leading_and_trailing_silence() -> None:
    silent = _rest_bar()
    middle_silent = _rest_bar()
    right_note = _note_bar()
    left_note = _chord_bar()
    score = parsed_score(
        right_hand_bars=[silent, right_note, middle_silent, bar([])],
        left_hand_bars=[bar([]), bar([]), left_note, silent],
    )

    cleaned = trim_silent_edge_bars(score)

    assert cleaned.right_hand_bars == [right_note, middle_silent]
    assert cleaned.left_hand_bars == [bar([]), left_note]


def test_trim_silent_edge_bars_keeps_interior_silent_bars() -> None:
    interior_silent = _rest_bar()
    score = parsed_score(
        right_hand_bars=[_note_bar(), interior_silent, _note_bar()],
        left_hand_bars=[bar([]), bar([]), bar([])],
    )

    cleaned = trim_silent_edge_bars(score)

    assert cleaned.right_hand_bars == score.right_hand_bars
    assert cleaned.left_hand_bars == score.left_hand_bars


def test_trim_silent_edge_bars_returns_empty_score_when_all_bars_are_silent() -> None:
    score = parsed_score(
        right_hand_bars=[_rest_bar(), bar([])],
        left_hand_bars=[bar([]), _rest_bar()],
    )

    cleaned = trim_silent_edge_bars(score)

    assert cleaned.right_hand_bars == []
    assert cleaned.left_hand_bars == []


def test_deduplicate_simultaneous_pitches_removes_repeated_chord_pitches() -> None:
    score = parsed_score(
        right_hand_bars=[bar([chord_event(midi_pitches=[60, 64, 64, 67], duration=Fraction(1, 4), beat_offset=0)])],
        left_hand_bars=[bar([])],
    )

    cleaned = deduplicate_simultaneous_pitches(score)

    chord = cleaned.right_hand_bars[0].events[0]
    assert isinstance(chord, ParsedChord)
    assert chord.midi_pitches == [60, 64, 67]


def test_deduplicate_simultaneous_pitches_converts_single_pitch_chord_to_note() -> None:
    score = parsed_score(
        right_hand_bars=[bar([chord_event(midi_pitches=[60, 60], duration=Fraction(1, 4), beat_offset=0)])],
        left_hand_bars=[bar([])],
    )

    cleaned = deduplicate_simultaneous_pitches(score)

    note = cleaned.right_hand_bars[0].events[0]
    assert isinstance(note, ParsedNote)
    assert note.midi_pitch == 60


def test_deduplicate_simultaneous_pitches_collapses_exact_simultaneous_note_duplicates() -> None:
    score = parsed_score(
        right_hand_bars=[
            bar(
                [
                    note_event(midi_pitch=60, duration=Fraction(1, 4), beat_offset=0),
                    note_event(midi_pitch=60, duration=Fraction(1, 4), beat_offset=0),
                    note_event(midi_pitch=64, duration=Fraction(1, 4), beat_offset=0),
                ]
            )
        ],
        left_hand_bars=[bar([])],
    )

    cleaned = deduplicate_simultaneous_pitches(score)

    chord = cleaned.right_hand_bars[0].events[0]
    assert isinstance(chord, ParsedChord)
    assert chord.midi_pitches == [60, 64]


def test_deduplicate_simultaneous_pitches_preserves_same_pitch_with_different_durations() -> None:
    score = parsed_score(
        right_hand_bars=[
            bar(
                [
                    note_event(midi_pitch=60, duration=Fraction(1, 4), beat_offset=0),
                    note_event(midi_pitch=60, duration=Fraction(1, 8), beat_offset=0),
                ]
            )
        ],
        left_hand_bars=[bar([])],
    )

    cleaned = deduplicate_simultaneous_pitches(score)

    assert cleaned.right_hand_bars[0].events == score.right_hand_bars[0].events


def test_truncate_overlapping_events_preserves_later_onsets() -> None:
    score = parsed_score(
        right_hand_bars=[
            bar(
                [
                    chord_event(midi_pitches=[60, 64], duration=Fraction(1, 2), beat_offset=Fraction(0)),
                    note_event(midi_pitch=67, duration=Fraction(1, 2), beat_offset=Fraction(1, 4)),
                    note_event(midi_pitch=69, duration=Fraction(1, 2), beat_offset=Fraction(1, 2)),
                ]
            )
        ],
        left_hand_bars=[bar([])],
    )

    cleaned = truncate_overlapping_events(score)

    events = cleaned.right_hand_bars[0].events
    assert [(event.beat_offset, event.duration) for event in events] == [
        (Fraction(0), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(1, 4)),
        (Fraction(1, 2), Fraction(1, 2)),
    ]
    assert isinstance(events[0], ParsedChord)
    assert isinstance(events[1], ParsedNote)
    assert isinstance(events[2], ParsedNote)


def test_clean_parsed_score_truncates_overlaps_before_deduplicating() -> None:
    score = parsed_score(
        right_hand_bars=[
            bar(
                [
                    note_event(midi_pitch=60, duration=Fraction(1, 2), beat_offset=Fraction(0)),
                    note_event(midi_pitch=60, duration=Fraction(3, 4), beat_offset=Fraction(0)),
                    note_event(midi_pitch=64, duration=Fraction(1, 2), beat_offset=Fraction(1, 4)),
                ]
            )
        ],
        left_hand_bars=[bar([])],
    )

    cleaned = clean_parsed_score(score)

    events = cleaned.right_hand_bars[0].events
    assert len(events) == 2
    assert isinstance(events[0], ParsedNote)
    assert events[0].midi_pitch == 60
    assert events[0].duration == Fraction(1, 4)


def test_clean_parsed_score_deduplicates_and_trims_silent_edges() -> None:
    score = parsed_score(
        right_hand_bars=[
            _rest_bar(),
            bar([chord_event(midi_pitches=[60, 60], duration=Fraction(1, 4), beat_offset=0)]),
            _rest_bar(),
        ],
        left_hand_bars=[bar([]), bar([]), bar([])],
    )

    cleaned = clean_parsed_score(score)

    assert len(cleaned.right_hand_bars) == 1
    assert isinstance(cleaned.right_hand_bars[0].events[0], ParsedNote)
