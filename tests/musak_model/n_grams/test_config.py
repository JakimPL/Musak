from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError

from musak_model.n_grams.config import NGramAnalysisConfig

type ConfigValue = float | int | tuple[Fraction, ...] | tuple[int, ...]


@dataclass(frozen=True)
class InvalidConfigCase:
    name: str
    overrides: dict[str, ConfigValue]
    expected_message: str


def test_n_gram_analysis_config_loads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "n_grams.yml"
    config_path.write_text(
        "\n".join(
            [
                "min_n: 2",
                "max_n: 4",
                "limit_per_group: 10",
                "workers: 3",
                "batch_size: 64",
                "figure_common_mass_threshold: 0.75",
                "rhythm_min_n: 1",
                "rhythm_max_n: 3",
                "grid_alignment_denominators:",
                "  - 1",
                "  - 2",
                "  - 4",
                "strong_beat_offsets:",
                "  - 0",
                '  - "1/2"',
            ]
        ),
        encoding="utf-8",
    )

    config = NGramAnalysisConfig.load(config_path)

    assert config.min_n == 2
    assert config.max_n == 4
    assert config.limit_per_group == 10
    assert config.workers == 3
    assert config.batch_size == 64
    assert config.figure_common_mass_threshold == 0.75
    assert config.rhythm_min_n == 1
    assert config.rhythm_max_n == 3
    assert config.grid_alignment_denominators == (1, 2, 4)
    assert config.strong_beat_offsets == (Fraction(0), Fraction(1, 2))


def test_n_gram_analysis_config_defaults_comparison_parameters() -> None:
    config = NGramAnalysisConfig(min_n=2, max_n=5, workers=1, batch_size=1)

    assert config.figure_common_mass_threshold == 0.80
    assert config.rhythm_min_n == 2
    assert config.rhythm_max_n == 4
    assert config.grid_alignment_denominators == (1, 2, 4, 8, 16)
    assert config.strong_beat_offsets == (Fraction(0),)


def test_n_gram_analysis_config_rejects_invalid_range() -> None:
    with pytest.raises(ValidationError, match="max_n must be greater than or equal to min_n"):
        NGramAnalysisConfig(min_n=5, max_n=2, workers=1, batch_size=1)


def test_n_gram_analysis_config_rejects_invalid_rhythm_range() -> None:
    with pytest.raises(ValidationError, match="rhythm_max_n must be greater than or equal to rhythm_min_n"):
        NGramAnalysisConfig(min_n=2, max_n=5, rhythm_min_n=3, rhythm_max_n=2, workers=1, batch_size=1)


def test_n_gram_analysis_config_rejects_non_positive_limit() -> None:
    with pytest.raises(ValidationError, match="limit_per_group"):
        NGramAnalysisConfig(min_n=2, max_n=5, limit_per_group=0, workers=1, batch_size=1)


def test_n_gram_analysis_config_rejects_non_positive_workers() -> None:
    with pytest.raises(ValidationError, match="workers"):
        NGramAnalysisConfig(min_n=2, max_n=5, workers=0, batch_size=1)


@pytest.mark.parametrize(
    "case",
    [
        InvalidConfigCase(
            name="invalid_common_mass_threshold",
            overrides={"figure_common_mass_threshold": 1.5},
            expected_message="figure_common_mass_threshold",
        ),
        InvalidConfigCase(
            name="empty_grid_denominators",
            overrides={"grid_alignment_denominators": ()},
            expected_message="grid_alignment_denominators must not be empty",
        ),
        InvalidConfigCase(
            name="non_positive_grid_denominators",
            overrides={"grid_alignment_denominators": (1, 0)},
            expected_message="grid_alignment_denominators must be positive",
        ),
        InvalidConfigCase(
            name="empty_strong_beat_offsets",
            overrides={"strong_beat_offsets": ()},
            expected_message="strong_beat_offsets must not be empty",
        ),
        InvalidConfigCase(
            name="negative_strong_beat_offsets",
            overrides={"strong_beat_offsets": (Fraction(-1, 4),)},
            expected_message="strong_beat_offsets must be non-negative",
        ),
    ],
    ids=lambda case: case.name,
)
def test_n_gram_analysis_config_rejects_invalid_comparison_parameters(case: InvalidConfigCase) -> None:
    payload: dict[str, ConfigValue] = {"min_n": 2, "max_n": 5, "workers": 1, "batch_size": 1}
    payload.update(case.overrides)
    with pytest.raises(ValidationError, match=case.expected_message):
        NGramAnalysisConfig.model_validate(payload)
