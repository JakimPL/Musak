import pytest
from scipy.special import expit

from musak_model.synthetic.fitting.accent import AccentMoments, fit_accent_config, fit_accent_overrides
from musak_model.synthetic.processes.accent import AccentFieldConfig
from musak_model.tokens.schema import Hand, ScaleType


def _default() -> AccentFieldConfig:
    return AccentFieldConfig(
        baseline_logit=-0.5,
        metric_gain=2.0,
        metric_exponent=1.0,
        envelope_basis_count=3,
        envelope_amplitude=0.5,
        envelope_decay=1.0,
    )


def test_fit_accent_config_reproduces_strong_and_weak_onset_rates() -> None:
    config = fit_accent_config(AccentMoments(strong_onset_rate=0.8, weak_onset_rate=0.2), default=_default())

    assert float(expit(config.baseline_logit)) == pytest.approx(0.2)
    assert float(expit(config.baseline_logit + config.metric_gain)) == pytest.approx(0.8)


def test_fit_accent_config_keeps_envelope_and_exponent_from_default() -> None:
    default = _default()

    config = fit_accent_config(AccentMoments(strong_onset_rate=0.6, weak_onset_rate=0.3), default=default)

    assert config.metric_exponent == default.metric_exponent
    assert config.envelope_basis_count == default.envelope_basis_count
    assert config.envelope_amplitude == default.envelope_amplitude
    assert config.envelope_decay == default.envelope_decay


def test_fit_accent_config_clamps_metric_gain_to_non_negative() -> None:
    config = fit_accent_config(AccentMoments(strong_onset_rate=0.2, weak_onset_rate=0.8), default=_default())

    assert config.metric_gain == 0.0


def test_fit_accent_config_handles_degenerate_rates() -> None:
    config = fit_accent_config(AccentMoments(strong_onset_rate=1.0, weak_onset_rate=0.0), default=_default())

    assert config.metric_gain >= 0.0
    assert config.baseline_logit < 0.0


def test_fit_accent_overrides_keys_each_group() -> None:
    overrides = fit_accent_overrides(
        {
            (ScaleType.MAJOR, Hand.RIGHT): AccentMoments(strong_onset_rate=0.8, weak_onset_rate=0.2),
            (ScaleType.MAJOR, Hand.LEFT): AccentMoments(strong_onset_rate=0.5, weak_onset_rate=0.4),
        },
        default=_default(),
    )

    by_group = {(override.scale_type, override.hand): override.config for override in overrides}
    assert float(expit(by_group[(ScaleType.MAJOR, Hand.RIGHT)].baseline_logit)) == pytest.approx(0.2)
    assert by_group[(ScaleType.MAJOR, Hand.LEFT)].metric_exponent == 1.0
