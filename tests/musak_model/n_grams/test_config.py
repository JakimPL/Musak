from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError

from musak_model.n_grams.config import (
    ExecutionConfig,
    FigureAnalysisConfig,
    NGramAnalysisConfig,
    RhythmAnalysisConfig,
)


@dataclass(frozen=True)
class InvalidRhythmCase:
    name: str
    grid_alignment_denominators: tuple[int, ...]
    strong_beat_offsets: tuple[Fraction, ...]
    expected_message: str


def test_n_gram_analysis_config_loads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "n_grams.yml"
    config_path.write_text(
        "\n".join(
            [
                "figure:",
                "  min_n: 2",
                "  max_n: 4",
                "  limit_per_group: 10",
                "  common_mass_threshold: 0.75",
                "rhythm:",
                "  min_n: 1",
                "  max_n: 3",
                "  grid_alignment_denominators:",
                "    - 1",
                "    - 2",
                "    - 4",
                "  strong_beat_offsets:",
                "    - 0",
                '    - "1/2"',
                "register:",
                "  arch_basis_count: 5",
                "execution:",
                "  workers: 3",
                "  batch_size: 64",
            ]
        ),
        encoding="utf-8",
    )

    config = NGramAnalysisConfig.load(config_path)

    assert config.figure.min_n == 2
    assert config.figure.max_n == 4
    assert config.figure.limit_per_group == 10
    assert config.figure.common_mass_threshold == 0.75
    assert config.rhythm.min_n == 1
    assert config.rhythm.max_n == 3
    assert config.rhythm.grid_alignment_denominators == (1, 2, 4)
    assert config.rhythm.strong_beat_offsets == (Fraction(0), Fraction(1, 2))
    assert config.register.arch_basis_count == 5
    assert config.execution.workers == 3
    assert config.execution.batch_size == 64


def test_figure_analysis_config_rejects_invalid_range() -> None:
    with pytest.raises(ValidationError, match="max_n must be greater than or equal to min_n"):
        FigureAnalysisConfig(min_n=5, max_n=2, limit_per_group=None, common_mass_threshold=0.8)


def test_rhythm_analysis_config_rejects_invalid_range() -> None:
    with pytest.raises(ValidationError, match="rhythm max_n must be greater than or equal to rhythm min_n"):
        RhythmAnalysisConfig(min_n=3, max_n=2, grid_alignment_denominators=(1,), strong_beat_offsets=(Fraction(0),))


def test_figure_analysis_config_rejects_non_positive_limit() -> None:
    with pytest.raises(ValidationError, match="limit_per_group"):
        FigureAnalysisConfig(min_n=2, max_n=5, limit_per_group=0, common_mass_threshold=0.8)


def test_figure_analysis_config_rejects_invalid_common_mass_threshold() -> None:
    with pytest.raises(ValidationError, match="common_mass_threshold"):
        FigureAnalysisConfig(min_n=2, max_n=5, limit_per_group=None, common_mass_threshold=1.5)


def test_execution_config_rejects_non_positive_workers() -> None:
    with pytest.raises(ValidationError, match="workers"):
        ExecutionConfig(workers=0, batch_size=1)


@pytest.mark.parametrize(
    "case",
    [
        InvalidRhythmCase(
            name="empty_grid_denominators",
            grid_alignment_denominators=(),
            strong_beat_offsets=(Fraction(0),),
            expected_message="grid_alignment_denominators must not be empty",
        ),
        InvalidRhythmCase(
            name="non_positive_grid_denominators",
            grid_alignment_denominators=(1, 0),
            strong_beat_offsets=(Fraction(0),),
            expected_message="grid_alignment_denominators must be positive",
        ),
        InvalidRhythmCase(
            name="empty_strong_beat_offsets",
            grid_alignment_denominators=(1,),
            strong_beat_offsets=(),
            expected_message="strong_beat_offsets must not be empty",
        ),
        InvalidRhythmCase(
            name="negative_strong_beat_offsets",
            grid_alignment_denominators=(1,),
            strong_beat_offsets=(Fraction(-1, 4),),
            expected_message="strong_beat_offsets must be non-negative",
        ),
    ],
    ids=lambda case: case.name,
)
def test_rhythm_analysis_config_rejects_invalid_values(case: InvalidRhythmCase) -> None:
    with pytest.raises(ValidationError, match=case.expected_message):
        RhythmAnalysisConfig(
            min_n=2,
            max_n=4,
            grid_alignment_denominators=case.grid_alignment_denominators,
            strong_beat_offsets=case.strong_beat_offsets,
        )
