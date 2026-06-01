from collections import Counter

from numpy.random import default_rng

from musak_model.synthetic.fitting.artifacts import FittedGeneratorConfig
from musak_model.synthetic.fitting.form.fit import FormFittingConfig, _otsu_bucket, fit_form_priors
from musak_model.synthetic.fitting.form.statistics import (
    ClosingKey,
    FormStatistics,
    HistogramKey,
    PhraseLengthKey,
    SegmentLengthKey,
)
from musak_model.synthetic.structure.form import FormSampler
from musak_model.tokens.schema import ScaleType

_MAJOR = ScaleType.MAJOR.value


def _bimodal_histogram(scale_type: str, *, low: int, high: int) -> Counter[HistogramKey]:
    histogram: Counter[HistogramKey] = Counter()
    for bucket in range(4):
        histogram[HistogramKey(scale_type, bucket)] += low
    for bucket in range(16, 20):
        histogram[HistogramKey(scale_type, bucket)] += high

    return histogram


def _period_statistics() -> FormStatistics:
    return FormStatistics(
        phrase_length_counts=Counter({PhraseLengthKey(_MAJOR, 4): 10, PhraseLengthKey(_MAJOR, 8): 4}),
        segment_length_counts=Counter({SegmentLengthKey(_MAJOR, 2): 12}),
        closing_counts=Counter(
            {
                ClosingKey(_MAJOR, False, "predominant>dominant"): 6,
                ClosingKey(_MAJOR, True, "dominant>tonic"): 8,
            }
        ),
        similarity_histogram=_bimodal_histogram(_MAJOR, low=30, high=30),
        best_match_histogram=Counter({HistogramKey(_MAJOR, 18): 12, HistogramKey(_MAJOR, 19): 8}),
    )


def test_fits_a_usable_period_prior() -> None:
    priors = fit_form_priors(_period_statistics(), config=FormFittingConfig.load())

    prior = priors[ScaleType.MAJOR]
    assert {choice.is_final for choice in prior.closings} == {True, False}
    assert prior.repeat_probability > 0.5

    form = FormSampler(prior).sample(bar_count=8, rng=default_rng(0))
    assert form.bar_count == 8
    assert form.phrases[-1].closing.terminal_function.value in {"tonic", "dominant", "predominant"}


def test_sparse_scale_falls_back_to_the_configured_prior() -> None:
    config = FormFittingConfig.load()
    statistics = FormStatistics(closing_counts=Counter({ClosingKey("harmonic_minor", True, "dominant>tonic"): 1}))

    priors = fit_form_priors(statistics, config=config)

    assert priors[ScaleType.HARMONIC_MINOR] == config.fallback_prior


def test_round_trips_through_fitted_generator_config() -> None:
    priors = fit_form_priors(_period_statistics(), config=FormFittingConfig.load())

    fitted = FittedGeneratorConfig(form_priors=priors)
    restored = FittedGeneratorConfig.model_validate_json(fitted.model_dump_json())

    assert restored.form_prior(ScaleType.MAJOR) == priors[ScaleType.MAJOR]


def test_otsu_bucket_finds_the_valley_between_modes() -> None:
    histogram = [40, 40, 0, 0, 0, 0, 0, 0, 40, 40]

    bucket = _otsu_bucket(histogram)

    assert bucket is not None
    assert 1 <= bucket <= 7
