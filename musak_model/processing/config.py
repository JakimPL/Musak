from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from musak_model.data.scale_matcher.config import ScaleMatcherConfig
from musak_model.paths import PROCESSING_CONFIG_PATH
from musak_shared.files import load_yaml_config


class ParsingProcessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workers: int = Field(gt=0)


class TokenizationProcessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workers: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    remove_segments_with_silent_bars: bool
    scale_matcher: ScaleMatcherConfig


class ProcessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parsing: ParsingProcessingConfig
    tokenization: TokenizationProcessingConfig

    @classmethod
    def load(cls, path: Path = PROCESSING_CONFIG_PATH) -> ProcessingConfig:
        parsed = load_yaml_config(path)
        return cls.model_validate(parsed)


def processing_config_with_overrides(
    config: ProcessingConfig,
    *,
    workers: int | None,
    tokenization_workers: int | None,
    tokenization_batch_size: int | None,
    remove_segments_with_silent_bars: bool | None,
    scale_match_support_score_margin: float | None,
    scale_match_selection_score_margin: float | None,
    scale_match_maximum_unexplained_weight_fraction: float | None,
    scale_match_maximum_explanation_pitch_class_count: int | None,
) -> ProcessingConfig:
    parsing_values = config.parsing.model_dump()
    if workers is not None:
        parsing_values["workers"] = workers

    tokenization_values = config.tokenization.model_dump()
    if tokenization_workers is not None:
        tokenization_values["workers"] = tokenization_workers
    if tokenization_batch_size is not None:
        tokenization_values["batch_size"] = tokenization_batch_size
    if remove_segments_with_silent_bars is not None:
        tokenization_values["remove_segments_with_silent_bars"] = remove_segments_with_silent_bars

    scale_matcher_values = config.tokenization.scale_matcher.model_dump()
    for key, value in {
        "support_score_margin": scale_match_support_score_margin,
        "selection_score_margin": scale_match_selection_score_margin,
        "maximum_unexplained_weight_fraction": scale_match_maximum_unexplained_weight_fraction,
        "maximum_explanation_pitch_class_count": scale_match_maximum_explanation_pitch_class_count,
    }.items():
        if value is not None:
            scale_matcher_values[key] = value

    tokenization_values["scale_matcher"] = ScaleMatcherConfig.model_validate(scale_matcher_values)

    return ProcessingConfig(
        parsing=ParsingProcessingConfig.model_validate(parsing_values),
        tokenization=TokenizationProcessingConfig.model_validate(tokenization_values),
    )
