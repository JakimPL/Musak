from pathlib import Path

import pytest

from musak_model.conditioning.config import ConditioningConfig


def test_conditioning_config_loads_yaml_and_derives_vocab_sizes(tmp_path: Path) -> None:
    config_path = tmp_path / "conditioning.yml"
    config_path.write_text(
        "\n".join(
            [
                "difficulty:",
                "  max_level: 5",
                "time_signature:",
                "  max_denominator: 4",
                "  relative_numerator_range: 2",
                "harmony:",
                "  enabled: true",
                "cfg_dropout_probability: 0.1",
            ]
        )
    )

    config = ConditioningConfig.load(config_path)

    assert config.num_difficulty_levels == 6
    assert config.num_scale_types == 3
    assert config.num_time_signatures == 11
    assert config.harmony.enabled is True


def test_conditioning_config_requires_semantic_fields() -> None:
    with pytest.raises(ValueError, match="difficulty"):
        ConditioningConfig.model_validate(
            {
                "time_signature": {"max_denominator": 4, "relative_numerator_range": 2},
                "harmony": {"enabled": True},
                "cfg_dropout_probability": 0.1,
            }
        )

    with pytest.raises(ValueError, match="time_signature"):
        ConditioningConfig.model_validate(
            {
                "difficulty": {"max_level": 5},
                "harmony": {"enabled": True},
                "cfg_dropout_probability": 0.1,
            }
        )

    with pytest.raises(ValueError, match="harmony"):
        ConditioningConfig.model_validate(
            {
                "difficulty": {"max_level": 5},
                "time_signature": {"max_denominator": 4, "relative_numerator_range": 2},
                "cfg_dropout_probability": 0.1,
            }
        )


def test_conditioning_config_rejects_derived_scalar_fields() -> None:
    with pytest.raises(ValueError, match="num_time_signatures"):
        ConditioningConfig.model_validate(
            {
                "difficulty": {"max_level": 5},
                "time_signature": {"max_denominator": 4, "relative_numerator_range": 2},
                "harmony": {"enabled": True},
                "num_time_signatures": 11,
                "cfg_dropout_probability": 0.1,
            }
        )
