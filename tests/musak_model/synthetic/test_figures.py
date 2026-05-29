from collections import Counter
from fractions import Fraction
from pathlib import Path
from random import Random

import pytest

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.io import write_figure_counts
from musak_model.synthetic.figures import (
    FigureVocabulary,
    load_figure_split_vocabulary,
    load_figure_vocabulary,
    resolve_figure_counts_path,
)
from musak_model.tokens.schema import Hand, ScaleType


def test_load_figure_vocabulary_reads_counts_as_figure_objects(tmp_path: Path) -> None:
    counts_path = _write_counts(tmp_path / "counts.parquet")

    vocabulary = load_figure_vocabulary(counts_path)

    assert vocabulary.unique_count == 2
    assert vocabulary.total_count == 5
    assert str(vocabulary.entries[0].figure) == "0(1) +1(1)"
    assert vocabulary.entries[0].group.scale_type == ScaleType.MAJOR
    assert vocabulary.entries[0].group.hand == Hand.RIGHT
    assert vocabulary.entries[0].group.n == 2


def test_load_figure_vocabulary_resolves_split_directory(tmp_path: Path) -> None:
    split_directory = tmp_path / "split" / "train"
    counts_path = split_directory / "all" / "counts.parquet"
    _write_counts(counts_path)

    vocabulary = load_figure_vocabulary(split_directory)

    assert vocabulary.total_count == 5


def test_load_figure_vocabulary_resolves_processed_encoded_directory(tmp_path: Path) -> None:
    encoded_directory = tmp_path / "encoded"
    counts_path = encoded_directory / "figure" / "all" / "counts.parquet"
    _write_counts(counts_path)

    assert resolve_figure_counts_path(encoded_directory) == counts_path


def test_load_figure_split_vocabulary_reads_training_split_artifact(tmp_path: Path) -> None:
    _write_counts(tmp_path / "abc123" / "validation" / "all" / "counts.parquet")

    vocabulary = load_figure_split_vocabulary(
        split_key="abc123",
        split_name="validation",
        artifact_root=tmp_path,
    )

    assert vocabulary.total_count == 5


def test_figure_vocabulary_filters_by_group_and_properties(tmp_path: Path) -> None:
    vocabulary = load_figure_vocabulary(_write_counts(tmp_path / "counts.parquet"))

    filtered = vocabulary.filter(
        scale_type=ScaleType.MAJOR,
        hand=Hand.RIGHT,
        n=2,
        monophonic=True,
        in_scale=True,
        min_count=3,
    )

    assert filtered.unique_count == 1
    assert str(filtered.entries[0].figure) == "0(1) +1(1)"


def test_figure_vocabulary_reports_count_weighted_length_distribution(tmp_path: Path) -> None:
    vocabulary = load_figure_vocabulary(_write_counts(tmp_path / "counts.parquet"))

    assert vocabulary.length_distribution() == {2: 0.6, 3: 0.4}


def test_figure_vocabulary_samples_with_commonness_bias() -> None:
    common = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    rare = FigureNGram(onsets=((((1, 0),), Fraction(1)),))
    vocabulary = FigureVocabulary.from_counts(
        {
            ScaleType.MAJOR: {
                Hand.RIGHT: {
                    1: Counter({common: 100, rare: 1}),
                }
            }
        }
    )

    sample = vocabulary.sample(rng=Random(0), commonness_bias=1.0)

    assert sample.figure == common


def test_figure_vocabulary_rejects_empty_sampling() -> None:
    with pytest.raises(ValueError, match="empty"):
        FigureVocabulary(entries=()).sample(rng=Random(0))


def _write_counts(path: Path) -> Path:
    first = FigureNGram(onsets=((((0, 0),), Fraction(1)), (((1, 0),), Fraction(1))))
    second = FigureNGram(onsets=((((0, 0),), Fraction(1)), (((2, 1),), Fraction(1)), (((4, 0),), Fraction(1))))
    write_figure_counts(
        {
            ScaleType.MAJOR: {
                Hand.RIGHT: {
                    2: Counter({first: 3}),
                    3: Counter({second: 2}),
                }
            }
        },
        path,
    )
    return path
