from musak_model.synthetic.calibration.config import CalibrationConfig


def test_default_config_loads() -> None:
    config = CalibrationConfig.load()

    assert config.samples_per_config > 0
    assert config.max_n >= config.min_n
    assert config.figure_lengths == tuple(range(config.min_n, config.max_n + 1))
    assert config.lambda_curve and config.lambda_harm and config.lambda_accent
