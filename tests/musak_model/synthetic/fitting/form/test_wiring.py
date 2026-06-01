from collections import Counter

from numpy.random import default_rng

from musak_model.synthetic.fitting.artifacts import FittedGeneratorConfig
from musak_model.synthetic.fitting.form.fit import FormFittingConfig, fit_form_priors
from musak_model.synthetic.fitting.form.statistics import ClosingKey, FormStatistics, PhraseLengthKey, SegmentLengthKey
from musak_model.synthetic.structure.form import FormSampler
from musak_model.tokens.schema import ScaleType

_MAJOR = ScaleType.MAJOR.value


def _statistics() -> FormStatistics:
    return FormStatistics(
        phrase_length_counts=Counter({PhraseLengthKey(_MAJOR, 4): 10}),
        segment_length_counts=Counter({SegmentLengthKey(_MAJOR, 2): 10}),
        closing_counts=Counter(
            {
                ClosingKey(_MAJOR, False, "predominant>dominant"): 6,
                ClosingKey(_MAJOR, True, "dominant>tonic"): 8,
            }
        ),
    )


def test_fitted_config_form_prior_drives_the_form_sampler() -> None:
    fitted = FittedGeneratorConfig(form_priors=fit_form_priors(_statistics(), config=FormFittingConfig.load()))
    restored = FittedGeneratorConfig.model_validate_json(fitted.model_dump_json())

    prior = restored.form_prior(ScaleType.MAJOR)
    assert prior is not None

    form = FormSampler(prior).sample(bar_count=8, rng=default_rng(0))
    assert form.bar_count == 8
    assert sum(phrase.bar_span for phrase in form.phrases) == 8
    assert sum(segment.bar_span for segment in form.segments) == 8


def test_missing_scale_returns_no_prior() -> None:
    fitted = FittedGeneratorConfig(form_priors=fit_form_priors(_statistics(), config=FormFittingConfig.load()))

    assert fitted.form_prior(ScaleType.MELODIC_MINOR) is None
