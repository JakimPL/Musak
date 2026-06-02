from fractions import Fraction

from musak_model.generation.constraints import GenerationConstraints
from musak_model.generation.coordinates import (
    decoder_input_coordinates_from_token_ids,
    decoder_input_coordinates_from_tokens,
)
from musak_model.tokens.duration import (
    DurationVocabulary,
    duration_fraction_to_ticks,
    duration_tick_denominator,
)
from musak_model.tokens.factorized import hand_to_attribute_id
from musak_model.tokens.schema import BarToken, Hand, HandToken, JoinWithPreviousToken, NoteToken, RestToken, Token
from musak_model.tokens.vocabulary import TokenVocabulary


def test_decoder_coordinates_track_active_hand_and_bar_relative_time(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    half_id = duration_vocabulary.fraction_to_id(Fraction(1, 2))
    denominator = duration_tick_denominator(duration_vocabulary)
    quarter_ticks = duration_fraction_to_ticks(Fraction(1, 4), denominator=denominator)
    half_ticks = duration_fraction_to_ticks(Fraction(1, 2), denominator=denominator)
    whole_ticks = duration_fraction_to_ticks(Fraction(1, 1), denominator=denominator)

    coordinates = decoder_input_coordinates_from_tokens(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            RestToken(duration_id=quarter_id),
            HandToken(hand=Hand.LEFT),
            RestToken(duration_id=half_id),
            BarToken(),
            HandToken(hand=Hand.RIGHT),
        ],
        constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=2),
        duration_vocabulary=duration_vocabulary,
        duration_tick_denominator=denominator,
    )

    assert coordinates.bar_relative_ticks == (0, 0, quarter_ticks, half_ticks, 0, half_ticks, 0, 0)
    assert coordinates.bar_indices == (0, 0, 0, 0, 0, 0, 1, 1)
    assert coordinates.bar_duration_ticks == (whole_ticks,) * 8
    assert coordinates.active_hand_ids == (
        hand_to_attribute_id(Hand.RIGHT),
        hand_to_attribute_id(Hand.RIGHT),
        hand_to_attribute_id(Hand.RIGHT),
        hand_to_attribute_id(Hand.RIGHT),
        hand_to_attribute_id(Hand.LEFT),
        hand_to_attribute_id(Hand.LEFT),
        hand_to_attribute_id(Hand.LEFT),
        hand_to_attribute_id(Hand.RIGHT),
    )


def test_decoder_coordinates_rewind_same_hand_joined_chord_cursor(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    denominator = duration_tick_denominator(duration_vocabulary)
    quarter_ticks = duration_fraction_to_ticks(Fraction(1, 4), denominator=denominator)
    half_ticks = duration_fraction_to_ticks(Fraction(1, 2), denominator=denominator)

    coordinates = decoder_input_coordinates_from_tokens(
        [
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=quarter_id),
            JoinWithPreviousToken(),
        ],
        constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1),
        duration_vocabulary=duration_vocabulary,
        duration_tick_denominator=denominator,
    )

    assert coordinates.bar_relative_ticks == (0, quarter_ticks, half_ticks, quarter_ticks)


def test_decoder_coordinates_from_token_ids_matches_tokens(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    tokens: list[Token] = [
        HandToken(hand=Hand.LEFT),
        RestToken(duration_id=quarter_id),
    ]
    denominator = duration_tick_denominator(duration_vocabulary)

    from_ids = decoder_input_coordinates_from_token_ids(
        token_vocabulary.encode(tokens),
        constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1),
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        duration_tick_denominator=denominator,
    )
    from_tokens = decoder_input_coordinates_from_tokens(
        tokens,
        constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1),
        duration_vocabulary=duration_vocabulary,
        duration_tick_denominator=denominator,
    )

    assert from_ids == from_tokens


def test_decoder_coordinates_use_declared_bar_durations(duration_vocabulary: DurationVocabulary) -> None:
    denominator = duration_tick_denominator(duration_vocabulary)
    half_ticks = duration_fraction_to_ticks(Fraction(1, 2), denominator=denominator)
    whole_ticks = duration_fraction_to_ticks(Fraction(1, 1), denominator=denominator)

    coordinates = decoder_input_coordinates_from_tokens(
        [BarToken()],
        constraints=GenerationConstraints(
            time_numerator=4,
            time_denominator=4,
            bar_count=2,
            bar_durations=(Fraction(1, 2), Fraction(1, 1)),
        ),
        duration_vocabulary=duration_vocabulary,
        duration_tick_denominator=denominator,
    )

    assert coordinates.bar_relative_ticks == (0, 0)
    assert coordinates.bar_indices == (0, 1)
    assert coordinates.bar_duration_ticks == (half_ticks, whole_ticks)


def test_decoder_coordinates_tolerate_early_bar_tokens(duration_vocabulary: DurationVocabulary) -> None:
    denominator = duration_tick_denominator(duration_vocabulary)

    coordinates = decoder_input_coordinates_from_tokens(
        [BarToken(), BarToken()],
        constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1),
        duration_vocabulary=duration_vocabulary,
        duration_tick_denominator=denominator,
    )

    assert coordinates.bar_relative_ticks == (0, 0, 0)
