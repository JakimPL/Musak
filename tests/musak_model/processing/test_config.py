from pathlib import Path

import pytest
from pydantic import ValidationError

from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.processing.config import (
    ParsingProcessingConfig,
    ProcessingConfig,
    TokenizationProcessingConfig,
    processing_config_with_overrides,
)


def test_processing_config_loads_parsing_and_tokenization_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "processing.yml"
    config_path.write_text(
        "\n".join(
            [
                "parsing:",
                "  workers: 2",
                "tokenization:",
                "  remove_segments_with_silent_bars: true",
                "  scale_matcher:",
                "    support_score_margin: 0.08",
                "    selection_score_margin: 0.03",
                "    maximum_unexplained_weight_fraction: 0.10",
                "    maximum_explanation_pitch_class_count: 9",
                "  workers: 3",
                "  batch_size: 5",
            ]
        ),
        encoding="utf-8",
    )

    config = ProcessingConfig.load(config_path)

    assert config.parsing.workers == 2
    assert config.tokenization.workers == 3
    assert config.tokenization.batch_size == 5
    assert config.tokenization.remove_segments_with_silent_bars is True


def test_parsing_processing_config_rejects_zero_workers() -> None:
    with pytest.raises(ValidationError, match="workers"):
        ParsingProcessingConfig(workers=0)


def test_tokenization_processing_config_rejects_zero_workers() -> None:
    with pytest.raises(ValidationError, match="workers"):
        TokenizationProcessingConfig(
            workers=0,
            batch_size=1,
            remove_segments_with_silent_bars=True,
            scale_matcher=ScaleMatcherConfig(
                support_score_margin=0.08,
                selection_score_margin=0.03,
                maximum_unexplained_weight_fraction=0.10,
                maximum_explanation_pitch_class_count=9,
            ),
        )


def test_processing_config_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValidationError, match="workers"):
        ProcessingConfig(
            parsing=ParsingProcessingConfig(workers=0),
            tokenization=TokenizationProcessingConfig(
                workers=1,
                batch_size=1,
                remove_segments_with_silent_bars=True,
                scale_matcher=ScaleMatcherConfig(
                    support_score_margin=0.08,
                    selection_score_margin=0.03,
                    maximum_unexplained_weight_fraction=0.10,
                    maximum_explanation_pitch_class_count=9,
                ),
            ),
        )


def test_processing_config_overrides_keep_yaml_values_by_default() -> None:
    config = ProcessingConfig.load()

    updated = processing_config_with_overrides(
        config,
        workers=None,
        tokenization_workers=4,
        tokenization_batch_size=6,
        remove_segments_with_silent_bars=False,
        scale_match_support_score_margin=None,
        scale_match_selection_score_margin=None,
        scale_match_maximum_unexplained_weight_fraction=None,
        scale_match_maximum_explanation_pitch_class_count=None,
    )

    assert updated.parsing.workers == config.parsing.workers
    assert updated.tokenization.workers == 4
    assert updated.tokenization.batch_size == 6
    assert updated.tokenization.remove_segments_with_silent_bars is False
    assert (
        updated.tokenization.scale_matcher.support_score_margin
        == config.tokenization.scale_matcher.support_score_margin
    )
