from fractions import Fraction

from musak_model.data.schema import ParsedBar, ParsedNote
from musak_model.decoder.piano_roll import PianoRollEvent, parsed_score_to_piano_roll_events
from musak_model.tokens.schema import Hand
from musak_model.validation.roundtrip import compare_parsed_score_to_events, compare_parsed_scores
from tests.data.fixtures import bar, chord_event, note_event, parsed_score


def test_compare_parsed_scores_reports_exact_match() -> None:
    score = parsed_score(
        right_hand_bars=[bar([note_event(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
        left_hand_bars=[bar([note_event(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(1, 4))])],
    )

    metrics = compare_parsed_scores(score, score)

    assert metrics.reference_note_count == 2
    assert metrics.decoded_note_count == 2
    assert metrics.exact_f1 == 1.0
    assert metrics.onset_pitch_f1 == 1.0
    assert metrics.time_signature_matches is True
    assert metrics.bar_count_matches is True


def test_compare_parsed_score_to_events_reports_duration_mismatch() -> None:
    score = parsed_score(
        right_hand_bars=[bar([note_event(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0))])],
        left_hand_bars=[bar([])],
    )
    decoded_events = [
        event.model_copy(update={"duration": Fraction(1, 8)}) for event in parsed_score_to_piano_roll_events(score)
    ]

    metrics = compare_parsed_score_to_events(score, decoded_events)

    assert metrics.exact_match_count == 0
    assert metrics.onset_pitch_match_count == 1
    assert metrics.mean_duration_error == Fraction(1, 8)
    assert metrics.max_duration_error == Fraction(1, 8)


def test_compare_parsed_score_to_events_reports_duplicate_pitch_onsets() -> None:
    score = parsed_score(
        right_hand_bars=[bar([chord_event(midi_pitches=[60, 64], duration=Fraction(1, 4), beat_offset=0)])],
        left_hand_bars=[bar([])],
    )
    duplicate_events = parsed_score_to_piano_roll_events(score) + [
        PianoRollEvent(hand=Hand.RIGHT, midi_pitch=60, start=Fraction(0), duration=Fraction(1, 4))
    ]

    metrics = compare_parsed_score_to_events(score, duplicate_events)

    assert metrics.reference_duplicate_pitch_onsets == 0
    assert metrics.decoded_duplicate_pitch_onsets == 1
    assert metrics.decoded_note_count == 3


def test_compare_parsed_scores_reports_bar_and_time_signature_mismatch() -> None:
    reference = parsed_score(
        right_hand_bars=[bar([note_event(midi_pitch=60, duration=Fraction(1, 4), beat_offset=0)]), bar([])],
        left_hand_bars=[bar([]), bar([])],
    )
    decoded = parsed_score(
        right_hand_bars=[
            ParsedBar(
                time_numerator=3,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=0)],
            )
        ],
        left_hand_bars=[ParsedBar(time_numerator=3, time_denominator=4, key_fifths=0, events=[])],
        time_numerator=3,
        time_denominator=4,
    )

    metrics = compare_parsed_scores(reference, decoded)

    assert metrics.bar_count_matches is False
    assert metrics.time_signature_matches is False
