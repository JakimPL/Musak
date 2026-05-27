from dataclasses import dataclass
from pathlib import Path

from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.profile.artifacts import FigureArtifactPaths, figure_artifact_paths
from musak_model.n_grams.profile.io import (
    read_figure_counts_csv,
    read_figure_profile,
    read_figure_sample_counts_jsonl,
)
from musak_model.n_grams.profile.rhythm.loading import RhythmProfileArtifacts, load_rhythm_profile_artifacts
from musak_model.n_grams.profile.schema import FigureProfile, FigureSampleCounts
from musak_model.processing.paths import ProcessedDatasetPaths
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary


@dataclass(frozen=True)
class FigureProfileArtifacts:
    paths: FigureArtifactPaths
    profile: FigureProfile
    counts_by_scale: FigureNGramCountsByScale
    sample_counts: tuple[FigureSampleCounts, ...]
    rhythm: RhythmProfileArtifacts | None


def load_figure_profile_artifacts(
    encoded_directory: Path,
    *,
    required: bool = False,
) -> FigureProfileArtifacts | None:
    paths = figure_artifact_paths(encoded_directory)
    missing_paths = _missing_artifact_paths(paths)
    if missing_paths:
        if not required and len(missing_paths) == len(_expected_artifact_paths(paths)):
            return None

        raise FileNotFoundError(
            f"Figure profile artifacts are incomplete under {paths.root_directory}; missing: "
            f"{', '.join(path.as_posix() for path in missing_paths)}"
        )

    profile = read_figure_profile(paths.profile_path)
    counts_by_scale = read_figure_counts_csv(paths.counts_path)
    sample_counts = tuple(read_figure_sample_counts_jsonl(paths.by_sample_path))
    rhythm_artifacts = load_rhythm_profile_artifacts(paths)
    _validate_artifact_consistency(profile=profile, sample_counts=sample_counts, paths=paths)
    return FigureProfileArtifacts(
        paths=paths,
        profile=profile,
        counts_by_scale=counts_by_scale,
        sample_counts=sample_counts,
        rhythm=rhythm_artifacts,
    )


def figure_profile_encoded_directory(
    *,
    processed_root: Path,
    dataset_root: Path,
    tokenization_config: TokenizationConfig,
) -> Path:
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    return paths.encoded_directory(snapshot.tokenizer_hash)


def load_processed_figure_profile_artifacts(
    *,
    processed_root: Path,
    dataset_root: Path,
    tokenization_config: TokenizationConfig,
    required: bool = False,
) -> FigureProfileArtifacts | None:
    encoded_directory = figure_profile_encoded_directory(
        processed_root=processed_root,
        dataset_root=dataset_root,
        tokenization_config=tokenization_config,
    )
    return load_figure_profile_artifacts(encoded_directory, required=required)


def _expected_artifact_paths(paths: FigureArtifactPaths) -> tuple[Path, ...]:
    return (
        paths.config_path,
        paths.counts_path,
        paths.profile_path,
        paths.by_sample_path,
    )


def _missing_artifact_paths(paths: FigureArtifactPaths) -> tuple[Path, ...]:
    return tuple(path for path in _expected_artifact_paths(paths) if not path.exists())


def _validate_artifact_consistency(
    *,
    profile: FigureProfile,
    sample_counts: tuple[FigureSampleCounts, ...],
    paths: FigureArtifactPaths,
) -> None:
    expected_sample_count = profile.metadata.sample_count
    if len(sample_counts) != expected_sample_count:
        raise ValueError(
            f"Figure profile sample_count={expected_sample_count} does not match "
            f"{len(sample_counts)} records in {paths.by_sample_path}"
        )

    sample_indices = tuple(sample_count.sample_index for sample_count in sample_counts)
    expected_indices = tuple(range(expected_sample_count))
    if sample_indices != expected_indices:
        raise ValueError(
            f"Figure sample counts in {paths.by_sample_path} must have contiguous sample_index values "
            f"from 0 to {expected_sample_count - 1}"
        )
