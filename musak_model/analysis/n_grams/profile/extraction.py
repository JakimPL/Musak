import shutil
from dataclasses import dataclass
from pathlib import Path

from musak_model.analysis.n_grams.config import NGramAnalysisConfig
from musak_model.analysis.n_grams.figure.encoded import count_encoded_exercises_figure_ngrams
from musak_model.analysis.n_grams.profile.artifacts import FigureArtifactPaths, figure_artifact_paths
from musak_model.analysis.n_grams.profile.builder import build_figure_profile
from musak_model.analysis.n_grams.profile.io import (
    figure_count_records,
    write_figure_count_csv,
    write_figure_counts_csv,
    write_figure_profile,
)
from musak_model.analysis.n_grams.profile.schema import FigureProfileMetadata
from musak_model.processing.io import load_encoded_jsonl, load_tokenizer_snapshot_json
from musak_model.processing.paths import ENCODED_JSONL_NAME, TOKENIZER_SNAPSHOT_NAME
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary


@dataclass(frozen=True)
class FigureExtractionResult:
    artifact_paths: FigureArtifactPaths
    encoded_sample_count: int
    profile_group_count: int
    extra_output_path: Path | None


def extract_figure_artifacts(
    *,
    encoded_dir: Path,
    analysis_config_path: Path,
    output_path: Path | None,
    show_progress: bool,
) -> FigureExtractionResult:
    config = NGramAnalysisConfig.load(analysis_config_path)
    artifact_paths = figure_artifact_paths(encoded_dir)
    tokenizer_snapshot_path = encoded_dir / TOKENIZER_SNAPSHOT_NAME
    encoded_jsonl_path = encoded_dir / ENCODED_JSONL_NAME
    snapshot = load_tokenizer_snapshot_json(tokenizer_snapshot_path)
    tokenization_config = TokenizationConfig.model_validate(snapshot.tokenization_config)
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    samples = load_encoded_jsonl(encoded_jsonl_path)
    counts = count_encoded_exercises_figure_ngrams(
        samples,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=config.min_n,
        max_n=config.max_n,
        workers=config.workers,
        batch_size=config.batch_size,
        show_progress=show_progress,
    )
    profile = build_figure_profile(
        counts,
        FigureProfileMetadata(
            min_n=config.min_n,
            max_n=config.max_n,
            sample_count=len(samples),
        ),
    )
    write_figure_counts_csv(counts, artifact_paths.counts_path)
    write_figure_profile(profile, artifact_paths.profile_path)
    copy_analysis_config(analysis_config_path, artifact_paths.config_path)
    if output_path is not None:
        records = figure_count_records(counts, limit_per_group=config.limit_per_group)
        write_figure_count_csv(records, output_path)

    return FigureExtractionResult(
        artifact_paths=artifact_paths,
        encoded_sample_count=len(samples),
        profile_group_count=len(profile.groups),
        extra_output_path=output_path,
    )


def copy_analysis_config(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == target_path.resolve():
        return

    shutil.copyfile(source_path, target_path)
