from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from musak_model.synthetic.base_durations import BaseDurationDistribution, load_base_duration_distribution
from musak_model.synthetic.figures import (
    AnchoredFigureVocabulary,
    FigureVocabulary,
    load_anchored_figure_vocabulary,
    load_figure_vocabulary,
)
from musak_model.synthetic.fitting.artifacts import FittedGeneratorConfig, resolve_fitted_generator_config_path
from musak_model.synthetic.fitting.figure_by_chord import load_figure_by_chord_model
from musak_model.synthetic.substitution import FigureByChordModel
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary


@dataclass(frozen=True)
class SyntheticInputs:
    figure_vocabulary: FigureVocabulary
    anchored_figure_vocabulary: AnchoredFigureVocabulary
    base_duration_distribution: BaseDurationDistribution
    duration_vocabulary: DurationVocabulary
    fitted: FittedGeneratorConfig
    figure_by_chord_model: FigureByChordModel


def load_synthetic_inputs(figure_directory: Path) -> SyntheticInputs:
    return SyntheticInputs(
        figure_vocabulary=load_figure_vocabulary(figure_directory),
        anchored_figure_vocabulary=load_anchored_figure_vocabulary(figure_directory),
        base_duration_distribution=load_base_duration_distribution(figure_directory),
        duration_vocabulary=DurationVocabulary(TokenizationConfig.load()),
        fitted=_load_fitted_generator_config(figure_directory),
        figure_by_chord_model=load_figure_by_chord_model(figure_directory),
    )


def _load_fitted_generator_config(figure_directory: Path) -> FittedGeneratorConfig:
    path = resolve_fitted_generator_config_path(figure_directory)
    return FittedGeneratorConfig.read(path) if path is not None else FittedGeneratorConfig()
