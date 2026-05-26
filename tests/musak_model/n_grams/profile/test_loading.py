from collections import Counter
from fractions import Fraction
from pathlib import Path

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.artifacts import FigureArtifactPaths, figure_artifact_paths
from musak_model.n_grams.profile.builder import build_figure_profile, build_figure_sample_counts
from musak_model.n_grams.profile.io import write_figure_profile, write_figure_sample_counts_jsonl
from musak_model.n_grams.profile.loading import (
    figure_profile_encoded_directory,
    load_figure_profile_artifacts,
    load_processed_figure_profile_artifacts,
)
from musak_model.n_grams.profile.schema import FigureProfileMetadata
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary


def test_load_figure_profile_artifacts_returns_none_when_optional_artifacts_are_missing(tmp_path: Path) -> None:
    assert load_figure_profile_artifacts(tmp_path / "encoded") is None


def test_load_figure_profile_artifacts_requires_complete_canonical_artifacts(tmp_path: Path) -> None:
    encoded_directory = tmp_path / "encoded"
    paths = figure_artifact_paths(encoded_directory)
    paths.config_path.parent.mkdir(parents=True)
    paths.config_path.write_text("min_n: 1\n", encoding="utf-8")

    try:
        load_figure_profile_artifacts(encoded_directory)
    except FileNotFoundError as error:
        message = str(error)
    else:
        raise AssertionError("expected incomplete artifacts to raise FileNotFoundError")

    assert "profile.json" in message
    assert "counts.csv" in message
    assert "by_sample.jsonl" in message


def test_load_figure_profile_artifacts_required_missing_artifacts_raise(tmp_path: Path) -> None:
    try:
        load_figure_profile_artifacts(tmp_path / "encoded", required=True)
    except FileNotFoundError as error:
        message = str(error)
    else:
        raise AssertionError("expected required artifacts to raise FileNotFoundError")

    assert "Figure profile artifacts are incomplete" in message


def test_load_figure_profile_artifacts_loads_and_validates_complete_artifacts(tmp_path: Path) -> None:
    encoded_directory = tmp_path / "encoded"
    profile = build_figure_profile(
        {
            ScaleType.MAJOR: {
                Hand.RIGHT: {
                    1: Counter({FigureNGram(onsets=((((0, 0),), Fraction(1)),)): 2}),
                }
            }
        },
        FigureProfileMetadata(min_n=1, max_n=1, sample_count=1),
    )
    sample_counts = build_figure_sample_counts(
        sample_index=0,
        scale_type=ScaleType.MAJOR,
        counts_by_hand={
            Hand.RIGHT: {
                1: Counter({FigureNGram(onsets=((((0, 0),), Fraction(1)),)): 2}),
            }
        },
    )
    paths = figure_artifact_paths(encoded_directory)
    _write_required_artifact_placeholders(paths)
    write_figure_profile(profile, paths.profile_path)
    write_figure_sample_counts_jsonl([sample_counts], paths.by_sample_path)

    artifacts = load_figure_profile_artifacts(encoded_directory)

    assert artifacts is not None
    assert artifacts.paths == paths
    assert artifacts.profile == profile
    assert artifacts.sample_counts == (sample_counts,)


def test_load_figure_profile_artifacts_validates_sample_count(tmp_path: Path) -> None:
    encoded_directory = tmp_path / "encoded"
    profile = build_figure_profile({}, FigureProfileMetadata(min_n=1, max_n=1, sample_count=1))
    paths = figure_artifact_paths(encoded_directory)
    _write_required_artifact_placeholders(paths)
    write_figure_profile(profile, paths.profile_path)
    write_figure_sample_counts_jsonl([], paths.by_sample_path)

    try:
        load_figure_profile_artifacts(encoded_directory)
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("expected inconsistent sample count to raise ValueError")

    assert "sample_count=1" in message


def test_figure_profile_encoded_directory_resolves_current_tokenizer_hash(
    tmp_path: Path,
    tokenization_config: TokenizationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    processed_root = tmp_path / "processed"
    dataset_root = tmp_path / "PDMX"
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    expected_dir = ProcessedDatasetPaths.from_dataset_root(
        processed_root=processed_root,
        dataset_root=dataset_root,
    ).encoded_directory(snapshot.tokenizer_hash)

    assert (
        figure_profile_encoded_directory(
            processed_root=processed_root,
            dataset_root=dataset_root,
            tokenization_config=tokenization_config,
        )
        == expected_dir
    )


def test_load_processed_figure_profile_artifacts_infers_encoded_directory(
    tmp_path: Path,
    tokenization_config: TokenizationConfig,
) -> None:
    processed_root = tmp_path / "processed"
    dataset_root = tmp_path / "PDMX"
    encoded_directory = figure_profile_encoded_directory(
        processed_root=processed_root,
        dataset_root=dataset_root,
        tokenization_config=tokenization_config,
    )
    profile = build_figure_profile({}, FigureProfileMetadata(min_n=1, max_n=1, sample_count=0))
    paths = figure_artifact_paths(encoded_directory)
    _write_required_artifact_placeholders(paths)
    write_figure_profile(profile, paths.profile_path)
    write_figure_sample_counts_jsonl([], paths.by_sample_path)

    artifacts = load_processed_figure_profile_artifacts(
        processed_root=processed_root,
        dataset_root=dataset_root,
        tokenization_config=tokenization_config,
    )

    assert artifacts is not None
    assert artifacts.paths == paths
    assert artifacts.profile == profile


def _write_required_artifact_placeholders(paths: FigureArtifactPaths) -> None:
    paths.config_path.parent.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text("min_n: 1\nmax_n: 1\n", encoding="utf-8")
    paths.counts_path.parent.mkdir(parents=True, exist_ok=True)
    paths.counts_path.write_text("scale_type,hand,n,count,figure\n", encoding="utf-8")
