from dataclasses import dataclass
from fractions import Fraction

import pytest

from musak_model.data.cleaning import (
    clean_parsed_score,
    deduplicate_simultaneous_pitches,
    normalize_simultaneous_event_durations,
    trim_silent_edge_bars,
    truncate_overlapping_events,
)
from musak_model.data.schema import ParsedChord, ParsedEvent, ParsedNote, ParsedRest
from tests.musak_model.data.fixtures import bar, chord_event, note_event, parsed_score, rest_event


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


@dataclass(frozen=True)
class SimultaneousDurationCase:
    name: str
    events: list[ParsedEvent]
    expected_pitched_durations: list[Fraction]
    expected_rest_durations: list[Fraction]

    def __str__(self) -> str:
        return self.name


class TestNormalizeSimultaneousEventDurations:
    CASES = [
        SimultaneousDurationCase(
            name="duration matching next onset is selected",
            events=[
                note_event(midi_pitch=60, duration=Fraction(1, 8), beat_offset=Fraction(0)),
                note_event(midi_pitch=64, duration=Fraction(1, 4), beat_offset=Fraction(0)),
                note_event(midi_pitch=67, duration=Fraction(1, 2), beat_offset=Fraction(0)),
                note_event(midi_pitch=69, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            ],
            expected_pitched_durations=[Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)],
            expected_rest_durations=[],
        ),
        SimultaneousDurationCase(
            name="next onset is selected when no duration matches",
            events=[
                note_event(midi_pitch=60, duration=Fraction(1, 8), beat_offset=Fraction(0)),
                note_event(midi_pitch=64, duration=Fraction(1, 2), beat_offset=Fraction(0)),
                note_event(midi_pitch=67, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            ],
            expected_pitched_durations=[Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)],
            expected_rest_durations=[],
        ),
        SimultaneousDurationCase(
            name="next onset can extend all simultaneous notes",
            events=[
                note_event(midi_pitch=60, duration=Fraction(1, 16), beat_offset=Fraction(0)),
                note_event(midi_pitch=64, duration=Fraction(1, 8), beat_offset=Fraction(0)),
                note_event(midi_pitch=67, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            ],
            expected_pitched_durations=[Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)],
            expected_rest_durations=[],
        ),
        SimultaneousDurationCase(
            name="next rest onset defines simultaneous note duration",
            events=[
                note_event(midi_pitch=60, duration=Fraction(1, 8), beat_offset=Fraction(0)),
                note_event(midi_pitch=64, duration=Fraction(1, 2), beat_offset=Fraction(0)),
                rest_event(duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            ],
            expected_pitched_durations=[Fraction(1, 4), Fraction(1, 4)],
            expected_rest_durations=[Fraction(1, 4)],
        ),
        SimultaneousDurationCase(
            name="longest duration is selected without later onset",
            events=[
                note_event(midi_pitch=60, duration=Fraction(1, 8), beat_offset=Fraction(0)),
                note_event(midi_pitch=64, duration=Fraction(1, 2), beat_offset=Fraction(0)),
            ],
            expected_pitched_durations=[Fraction(1, 2), Fraction(1, 2)],
            expected_rest_durations=[],
        ),
        SimultaneousDurationCase(
            name="single pitched event at onset is unchanged",
            events=[
                note_event(midi_pitch=60, duration=Fraction(1, 8), beat_offset=Fraction(0)),
                note_event(midi_pitch=64, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
            ],
            expected_pitched_durations=[Fraction(1, 8), Fraction(1, 4)],
            expected_rest_durations=[],
        ),
    ]

    @pytest.mark.parametrize("case", CASES, ids=str)
    def test_normalizes_same_onset_pitched_durations(self, case: SimultaneousDurationCase) -> None:
        score = parsed_score(right_hand_bars=[bar(case.events)], left_hand_bars=[bar([])])

        cleaned = normalize_simultaneous_event_durations(score)

        events = cleaned.right_hand_bars[0].events
        assert [event.duration for event in events if isinstance(event, ParsedNote | ParsedChord)] == (
            case.expected_pitched_durations
        )
        assert [event.duration for event in events if isinstance(event, ParsedRest)] == case.expected_rest_durations

    def test_normalizes_chords_and_notes_at_same_onset(self) -> None:
        score = parsed_score(
            right_hand_bars=[
                bar(
                    [
                        chord_event(midi_pitches=[60, 64], duration=Fraction(1, 8), beat_offset=Fraction(0)),
                        note_event(midi_pitch=67, duration=Fraction(1, 2), beat_offset=Fraction(0)),
                        note_event(midi_pitch=69, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
                    ]
                )
            ],
            left_hand_bars=[bar([])],
        )

        cleaned = normalize_simultaneous_event_durations(score)

        events = cleaned.right_hand_bars[0].events
        assert [event.duration for event in events] == [Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)]
        assert isinstance(events[0], ParsedChord)
        assert isinstance(events[1], ParsedNote)

    def test_normalizes_hands_independently(self) -> None:
        score = parsed_score(
            right_hand_bars=[
                bar(
                    [
                        note_event(midi_pitch=60, duration=Fraction(1, 8), beat_offset=Fraction(0)),
                        note_event(midi_pitch=64, duration=Fraction(1, 2), beat_offset=Fraction(0)),
                        note_event(midi_pitch=67, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
                    ]
                )
            ],
            left_hand_bars=[
                bar(
                    [
                        note_event(midi_pitch=48, duration=Fraction(1, 8), beat_offset=Fraction(0)),
                        note_event(midi_pitch=52, duration=Fraction(1, 2), beat_offset=Fraction(0)),
                    ]
                )
            ],
        )

        cleaned = normalize_simultaneous_event_durations(score)

        assert [event.duration for event in cleaned.right_hand_bars[0].events] == [
            Fraction(1, 4),
            Fraction(1, 4),
            Fraction(1, 4),
        ]
        assert [event.duration for event in cleaned.left_hand_bars[0].events] == [
            Fraction(1, 2),
            Fraction(1, 2),
        ]

    def test_clean_parsed_score_collapses_mixed_duration_same_onset_group_to_chord(self) -> None:
        score = parsed_score(
            right_hand_bars=[
                bar(
                    [
                        note_event(midi_pitch=60, duration=Fraction(1, 8), beat_offset=Fraction(0)),
                        note_event(midi_pitch=64, duration=Fraction(1, 2), beat_offset=Fraction(0)),
                        note_event(midi_pitch=67, duration=Fraction(1, 4), beat_offset=Fraction(1, 4)),
                    ]
                )
            ],
            left_hand_bars=[bar([])],
        )

        cleaned = clean_parsed_score(score)

        first_event = cleaned.right_hand_bars[0].events[0]
        assert isinstance(first_event, ParsedChord)
        assert first_event.midi_pitches == [60, 64]
        assert first_event.duration == Fraction(1, 4)


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
