from fractions import Fraction

import pytest

from musak_model.conditioning.harmony.alignment import (
    harmonic_plan_ids_from_decoder_coordinates,
    harmonic_plan_tensors_from_token_ids,
)
from musak_model.conditioning.harmony.schema import HarmonicPlanWindow
from musak_model.conditioning.harmony.vocabulary import (
    HARMONIC_PLAN_UNKNOWN_ID,
    harmonic_function_to_id,
    id_to_chord_change,
    id_to_harmonic_function,
    id_to_root_degree,
)
from musak_model.generation.constraints import GenerationConstraints
from musak_model.generation.coordinates import DecoderInputCoordinates
from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.tokens.duration import DurationVocabulary, duration_tick_denominator
from musak_model.tokens.schema import BarToken, NoteToken
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_shared.elements import HarmonicFunction


def test_harmonic_plan_alignment_uses_decoder_step_cursor(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    denominator = duration_tick_denominator(duration_vocabulary)
    windows = _tonic_to_dominant_windows()
    token_ids = token_vocabulary.encode(
        [
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=quarter_id),
        ]
    )

    tensors = harmonic_plan_tensors_from_token_ids(
        token_ids,
        windows=windows,
        constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1),
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        duration_tick_denominator=denominator,
    )

    assert tuple(id_to_harmonic_function(int(identifier)) for identifier in tensors.harmonic_function_ids) == (
        HarmonicFunction.TONIC,
        HarmonicFunction.TONIC,
        HarmonicFunction.DOMINANT,
        HarmonicFunction.DOMINANT,
    )
    assert tuple(id_to_root_degree(int(identifier)) for identifier in tensors.root_degree_ids) == (1, 1, 5, 5)
    assert tuple(id_to_chord_change(int(identifier)) for identifier in tensors.chord_change_ids) == (
        True,
        True,
        True,
        True,
    )


def test_harmonic_plan_alignment_uses_bar_duration_constraints() -> None:
    coordinates = DecoderInputCoordinates(
        bar_indices=(0, 1),
        bar_relative_ticks=(0, 0),
        bar_duration_ticks=(2, 4),
        active_hand_ids=(0, 0),
    )

    ids = harmonic_plan_ids_from_decoder_coordinates(
        _pickup_windows(),
        constraints=GenerationConstraints(
            time_numerator=4,
            time_denominator=4,
            bar_count=2,
            bar_durations=(Fraction(1, 2), Fraction(1)),
        ),
        coordinates=coordinates,
        duration_tick_denominator=4,
    )

    assert tuple(id_to_root_degree(item.root_degree_id) for item in ids) == (1, 5)


def test_harmonic_plan_alignment_returns_unknown_for_padding_and_gaps() -> None:
    coordinates = DecoderInputCoordinates(
        bar_indices=(-1, 0),
        bar_relative_ticks=(-1, 3),
        bar_duration_ticks=(1, 4),
        active_hand_ids=(-1, 0),
    )

    ids = harmonic_plan_ids_from_decoder_coordinates(
        _tonic_to_dominant_windows(),
        constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1),
        coordinates=coordinates,
        duration_tick_denominator=4,
    )

    assert ids[0].harmonic_function_id == HARMONIC_PLAN_UNKNOWN_ID
    assert ids[1].harmonic_function_id == harmonic_function_to_id(HarmonicFunction.DOMINANT)


def test_harmonic_plan_alignment_rejects_unknown_in_requested_span_when_strict() -> None:
    coordinates = DecoderInputCoordinates(
        bar_indices=(0,),
        bar_relative_ticks=(3,),
        bar_duration_ticks=(4,),
        active_hand_ids=(0,),
    )

    with pytest.raises(ValueError, match="in-span position"):
        harmonic_plan_ids_from_decoder_coordinates(
            (
                HarmonicPlanWindow(
                    start=Fraction(0),
                    end=Fraction(1, 2),
                    chord=Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
                ),
            ),
            constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1),
            coordinates=coordinates,
            duration_tick_denominator=4,
            strict_in_span=True,
        )


def test_harmonic_plan_alignment_rejects_overlapping_windows() -> None:
    coordinates = DecoderInputCoordinates(
        bar_indices=(0,),
        bar_relative_ticks=(0,),
        bar_duration_ticks=(4,),
        active_hand_ids=(0,),
    )

    with pytest.raises(ValueError, match="sorted and non-overlapping"):
        harmonic_plan_ids_from_decoder_coordinates(
            (
                HarmonicPlanWindow(
                    start=Fraction(0),
                    end=Fraction(1, 2),
                    chord=Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
                ),
                HarmonicPlanWindow(
                    start=Fraction(1, 4),
                    end=Fraction(3, 4),
                    chord=Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR),
                ),
            ),
            constraints=GenerationConstraints(time_numerator=4, time_denominator=4, bar_count=1),
            coordinates=coordinates,
            duration_tick_denominator=4,
        )


def _tonic_to_dominant_windows() -> tuple[HarmonicPlanWindow, ...]:
    return (
        HarmonicPlanWindow(
            start=Fraction(0),
            end=Fraction(1, 2),
            chord=Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
        ),
        HarmonicPlanWindow(
            start=Fraction(1, 2),
            end=Fraction(1),
            chord=Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR),
        ),
    )


def _pickup_windows() -> tuple[HarmonicPlanWindow, ...]:
    return (
        HarmonicPlanWindow(
            start=Fraction(0),
            end=Fraction(1, 2),
            chord=Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
        ),
        HarmonicPlanWindow(
            start=Fraction(1, 2),
            end=Fraction(3, 2),
            chord=Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR),
        ),
    )
