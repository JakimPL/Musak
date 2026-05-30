import pytest
from numpy.random import default_rng
from pydantic import ValidationError

from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.synthetic.processes.chord_track import (
    ChordTrackSampler,
    ChordTransitionModel,
    functional_transition_model,
    uniform_transition_model,
)
from musak_model.tokens.schema import ScaleType


def _chord(root_degree: int, quality: ChordQuality = ChordQuality.MAJOR) -> Chord:
    return Chord(root_degree=root_degree, root_accidental=0, quality=quality)


def _vocabulary() -> tuple[Chord, ...]:
    return (_chord(1), _chord(4), _chord(5))


def _major_diatonic_triads() -> tuple[Chord, ...]:
    return (
        _chord(1, ChordQuality.MAJOR),
        _chord(2, ChordQuality.MINOR),
        _chord(3, ChordQuality.MINOR),
        _chord(4, ChordQuality.MAJOR),
        _chord(5, ChordQuality.MAJOR),
        _chord(6, ChordQuality.MINOR),
        _chord(7, ChordQuality.DIMINISHED),
    )


def test_uniform_model_rows_and_initial_sum_to_one() -> None:
    model = uniform_transition_model(_vocabulary())

    assert pytest.approx(sum(model.initial_distribution.values())) == 1.0
    for row in model.transitions.values():
        assert pytest.approx(sum(row.values())) == 1.0


def test_uniform_model_self_bias_increases_diagonal_only() -> None:
    chords = _vocabulary()
    model = uniform_transition_model(chords, self_transition_bias=0.5)

    for source in chords:
        for destination in chords:
            if source == destination:
                assert model.transitions[source][destination] > 1.0 / len(chords)
            else:
                assert model.transitions[source][destination] < 1.0 / len(chords)


def test_sample_is_deterministic_for_a_given_seed() -> None:
    sampler = ChordTrackSampler(model=uniform_transition_model(_vocabulary()))

    first = sampler.sample(length=16, rng=default_rng(7))
    second = sampler.sample(length=16, rng=default_rng(7))

    assert first == second
    assert len(first) == 16


def test_full_self_transition_bias_locks_track_to_initial_chord() -> None:
    sampler = ChordTrackSampler(model=uniform_transition_model(_vocabulary(), self_transition_bias=1.0))

    track = sampler.sample(length=32, rng=default_rng(0))

    assert set(track) == {track[0]}


def test_zero_self_transition_bias_explores_full_vocabulary() -> None:
    chords = _vocabulary()
    sampler = ChordTrackSampler(model=uniform_transition_model(chords, self_transition_bias=0.0))

    track = sampler.sample(length=400, rng=default_rng(0))

    assert set(track) == set(chords)


def test_sample_rejects_non_positive_length() -> None:
    sampler = ChordTrackSampler(model=uniform_transition_model(_vocabulary()))

    with pytest.raises(ValueError, match="length"):
        sampler.sample(length=0, rng=default_rng(0))


def test_model_rejects_rows_that_do_not_sum_to_one() -> None:
    tonic = _chord(1)
    dominant = _chord(5)
    initial = {tonic: 1.0, dominant: 0.0}

    with pytest.raises(ValidationError):
        ChordTransitionModel(
            initial_distribution=initial,
            transitions={
                tonic: {tonic: 0.6, dominant: 0.3},
                dominant: {tonic: 0.5, dominant: 0.5},
            },
        )


def test_model_rejects_transitions_referencing_unknown_chords() -> None:
    tonic = _chord(1)
    dominant = _chord(5)
    intruder = _chord(2)

    with pytest.raises(ValidationError):
        ChordTransitionModel(
            initial_distribution={tonic: 0.5, dominant: 0.5},
            transitions={
                tonic: {tonic: 0.5, intruder: 0.5},
                dominant: {tonic: 0.5, dominant: 0.5},
            },
        )


def test_uniform_transition_model_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="chords"):
        uniform_transition_model(())

    with pytest.raises(ValueError, match="self_transition_bias"):
        uniform_transition_model(_vocabulary(), self_transition_bias=1.5)


def test_functional_model_rows_and_initial_sum_to_one() -> None:
    model = functional_transition_model(_major_diatonic_triads(), scale_type=ScaleType.MAJOR)

    assert pytest.approx(sum(model.initial_distribution.values())) == 1.0
    for row in model.transitions.values():
        assert pytest.approx(sum(row.values())) == 1.0


def test_functional_model_prefers_dominant_to_tonic_cadence() -> None:
    chords = _major_diatonic_triads()
    model = functional_transition_model(chords, scale_type=ScaleType.MAJOR, strength=0.8)

    dominant = _chord(5, ChordQuality.MAJOR)
    tonic = _chord(1, ChordQuality.MAJOR)
    supertonic = _chord(2, ChordQuality.MINOR)

    assert model.transitions[dominant][tonic] > model.transitions[dominant][supertonic]


def test_functional_model_prefers_predominant_to_dominant() -> None:
    chords = _major_diatonic_triads()
    model = functional_transition_model(chords, scale_type=ScaleType.MAJOR, strength=0.8)

    supertonic = _chord(2, ChordQuality.MINOR)
    dominant = _chord(5, ChordQuality.MAJOR)
    tonic = _chord(1, ChordQuality.MAJOR)

    assert model.transitions[supertonic][dominant] > model.transitions[supertonic][tonic]


def test_functional_model_initial_favors_tonic() -> None:
    chords = _major_diatonic_triads()
    model = functional_transition_model(chords, scale_type=ScaleType.MAJOR, strength=0.8)

    assert (
        model.initial_distribution[_chord(1, ChordQuality.MAJOR)]
        > model.initial_distribution[_chord(7, ChordQuality.DIMINISHED)]
    )


def test_functional_model_reduces_to_uniform_at_zero_strength() -> None:
    chords = _major_diatonic_triads()
    functional = functional_transition_model(
        chords, scale_type=ScaleType.MAJOR, strength=0.0, self_transition_bias=0.25
    )
    uniform = uniform_transition_model(chords, self_transition_bias=0.25)

    for chord in chords:
        assert functional.initial_distribution[chord] == pytest.approx(uniform.initial_distribution[chord])

    for source in chords:
        for destination in chords:
            assert functional.transitions[source][destination] == pytest.approx(
                uniform.transitions[source][destination]
            )


def test_functional_transition_model_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="chords"):
        functional_transition_model((), scale_type=ScaleType.MAJOR)

    with pytest.raises(ValueError, match="strength"):
        functional_transition_model(_major_diatonic_triads(), scale_type=ScaleType.MAJOR, strength=1.5)
