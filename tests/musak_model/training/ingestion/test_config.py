from pathlib import Path

import pytest

from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.training.ingestion.config import IngestionConfig


def test_load_ingestion_config_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "ingestion.yml"
    config_path.write_text(
        "\n".join(
            [
                "validation_fraction: 0.25",
                "split_seed: 99",
                "scale_matcher:",
                "  support_score_margin: 0.08",
                "  selection_score_margin: 0.03",
                "  maximum_unexplained_weight_fraction: 0.10",
                "  maximum_explanation_pitch_class_count: 9",
                "difficulty_labels:",
                "  sample: 3",
            ]
        )
    )

    config = IngestionConfig.load(config_path)

    assert config.validation_fraction == 0.25
    assert config.split_seed == 99
    assert config.difficulty_labels == {"sample": 3}


def test_ingestion_config_rejects_invalid_validation_fraction() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        IngestionConfig(
            validation_fraction=1.0,
            split_seed=17,
            scale_matcher=ScaleMatcherConfig(
                support_score_margin=0.08,
                selection_score_margin=0.03,
                maximum_unexplained_weight_fraction=0.10,
                maximum_explanation_pitch_class_count=9,
            ),
            difficulty_labels=None,
        )
