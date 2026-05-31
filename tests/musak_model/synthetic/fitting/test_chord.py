from collections import Counter

import pytest

from musak_model.harmony.decoding.candidates import spellable_candidates
from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.profile.chord.schema import INITIAL_CHORD_SOURCE, ChordTransitionKey, chord_to_key
from musak_model.synthetic.fitting.chord import fit_chord_transition_model
from musak_model.synthetic.processes.chord_track import functional_transition_model
from musak_model.tokens.schema import ScaleType

_TONIC = Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR)
_DOMINANT = Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR)
_SUPERTONIC = Chord(root_degree=2, root_accidental=0, quality=ChordQuality.MINOR)


def _major_prior():
    chords = tuple(
        candidate.chord for candidate in spellable_candidates(ChordVocabularyConfig.load(), scale_type=ScaleType.MAJOR)
    )
    return functional_transition_model(chords, scale_type=ScaleType.MAJOR, strength=0.7, self_transition_bias=0.0)


def test_fit_transition_model_sharpens_toward_observed_transitions() -> None:
    prior = _major_prior()
    counts = Counter(
        {
            ChordTransitionKey("major", INITIAL_CHORD_SOURCE, chord_to_key(_TONIC)): 10,
            ChordTransitionKey("major", chord_to_key(_TONIC), chord_to_key(_DOMINANT)): 20,
            ChordTransitionKey("major", chord_to_key(_DOMINANT), chord_to_key(_TONIC)): 18,
        }
    )

    model = fit_chord_transition_model(counts, scale_type=ScaleType.MAJOR, prior=prior, prior_count=1.0)

    assert model.transitions[_TONIC][_DOMINANT] > prior.transitions[_TONIC][_DOMINANT]
    assert model.initial_distribution[_TONIC] > prior.initial_distribution[_TONIC]
    assert all(probability > 0.0 for probability in model.transitions[_TONIC].values())  # prior floors every entry


def test_fit_transition_model_backs_off_to_prior_for_unobserved_source() -> None:
    prior = _major_prior()
    counts = Counter({ChordTransitionKey("major", chord_to_key(_TONIC), chord_to_key(_DOMINANT)): 5})

    model = fit_chord_transition_model(counts, scale_type=ScaleType.MAJOR, prior=prior, prior_count=1.0)

    assert model.transitions[_SUPERTONIC] == pytest.approx(prior.transitions[_SUPERTONIC])


def test_fit_transition_model_ignores_other_scale_types() -> None:
    prior = _major_prior()
    major_only = Counter({ChordTransitionKey("major", chord_to_key(_TONIC), chord_to_key(_DOMINANT)): 6})
    with_minor = Counter(major_only)
    with_minor[ChordTransitionKey("harmonic_minor", chord_to_key(_TONIC), chord_to_key(_DOMINANT))] = 99

    fitted_major_only = fit_chord_transition_model(major_only, scale_type=ScaleType.MAJOR, prior=prior, prior_count=1.0)
    fitted_with_minor = fit_chord_transition_model(with_minor, scale_type=ScaleType.MAJOR, prior=prior, prior_count=1.0)

    assert fitted_with_minor.transitions[_TONIC] == pytest.approx(fitted_major_only.transitions[_TONIC])
