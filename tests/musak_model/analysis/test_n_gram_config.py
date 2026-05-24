from pathlib import Path

import pytest
from pydantic import ValidationError

from musak_model.analysis.n_grams import NGramAnalysisConfig


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


def test_n_gram_analysis_config_rejects_invalid_range() -> None:
    with pytest.raises(ValidationError, match="max_n must be greater than or equal to min_n"):
        NGramAnalysisConfig(min_n=5, max_n=2, workers=1, batch_size=1)


def test_n_gram_analysis_config_rejects_non_positive_limit() -> None:
    with pytest.raises(ValidationError, match="limit_per_group"):
        NGramAnalysisConfig(min_n=2, max_n=5, limit_per_group=0, workers=1, batch_size=1)


def test_n_gram_analysis_config_rejects_non_positive_workers() -> None:
    with pytest.raises(ValidationError, match="workers"):
        NGramAnalysisConfig(min_n=2, max_n=5, workers=0, batch_size=1)
