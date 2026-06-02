from fractions import Fraction

from musak_model.conditioning.harmony.generation import FunctionalHarmonicPlanProvider, harmonic_plan_window_bounds
from musak_model.generation.constraints import GenerationConstraints
from musak_model.harmony.decoding import ChordDecoderConfig
from musak_model.harmony.diatonic import diatonic_triads
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.synthetic.fitting.chord import ChordFitConfig
from musak_model.tokens.schema import ScaleType


def _provider(*, resolution: int = 1) -> FunctionalHarmonicPlanProvider:
    return FunctionalHarmonicPlanProvider(
        chord_vocabulary=ChordVocabularyConfig.load(),
        chord_fit=ChordFitConfig(
            prior_count=1.0,
            functional_strength=0.8,
            self_transition_bias=0.25,
            figure_by_chord_limit=128,
        ),
        chord_decoding=ChordDecoderConfig(
            resolution=resolution,
            self_transition_bias=0.25,
            non_chord_penalty=1.0,
        ),
    )


def test_functional_provider_samples_deterministic_diatonic_plan() -> None:
    constraints = GenerationConstraints(
        time_numerator=4,
        time_denominator=4,
        bar_count=4,
        scale_type=ScaleType.MAJOR,
    )
    provider = _provider()

    first = provider.plan_windows(constraints=constraints, seed=7)
    second = provider.plan_windows(constraints=constraints, seed=7)

    assert first == second
    assert len(first) == 4
    assert tuple((window.start, window.end) for window in first) == (
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(2)),
        (Fraction(2), Fraction(3)),
        (Fraction(3), Fraction(4)),
    )
    assert {window.chord for window in first}.issubset(set(diatonic_triads(ScaleType.MAJOR)))


def test_harmonic_plan_window_bounds_respect_resolution_and_short_bars() -> None:
    bounds = harmonic_plan_window_bounds(
        constraints=GenerationConstraints(
            time_numerator=4,
            time_denominator=4,
            bar_count=2,
            bar_durations=(Fraction(1, 4), Fraction(1)),
        ),
        resolution=2,
    )

    assert bounds == (
        (Fraction(0), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(3, 4)),
        (Fraction(3, 4), Fraction(5, 4)),
    )
