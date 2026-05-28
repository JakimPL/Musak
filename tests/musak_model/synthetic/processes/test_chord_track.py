import pytest
from numpy.random import default_rng
from pydantic import ValidationError

from musak_model.synthetic.harmony.schema import Chord, ChordQuality
from musak_model.synthetic.processes.chord_track import (
    ChordTrackSampler,
    ChordTransitionModel,
    uniform_transition_model,
)


def _chord(root_degree: int, quality: ChordQuality = ChordQuality.MAJOR) -> Chord:
    return Chord(root_degree=root_degree, root_accidental=0, quality=quality)


def _vocabulary() -> tuple[Chord, ...]:
    return (_chord(1), _chord(4), _chord(5))


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
