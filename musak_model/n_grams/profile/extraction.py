import shutil
from dataclasses import dataclass
from pathlib import Path

from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.profile.artifacts import FigureArtifactPaths, figure_artifact_paths
from musak_model.n_grams.profile.streaming.orchestration import extract_streaming_figure_artifacts
from musak_model.processing.io import load_tokenizer_snapshot_json
from musak_model.processing.paths import ENCODED_JSONL_NAME, TOKENIZER_SNAPSHOT_NAME


@dataclass(frozen=True)
class FigureExtractionResult:
    artifact_paths: FigureArtifactPaths
    encoded_sample_count: int
    profile_group_count: int
    sample_profile_count: int
    extra_output_path: Path | None


def extract_figure_artifacts(
    *,
    encoded_directory: Path,
    analysis_config_path: Path,
    output_path: Path | None,
    show_progress: bool,
    overwrite: bool = False,
    resume: bool = False,
) -> FigureExtractionResult:
    config = NGramAnalysisConfig.load(analysis_config_path)
    artifact_paths = figure_artifact_paths(encoded_directory)
    tokenizer_snapshot_path = encoded_directory / TOKENIZER_SNAPSHOT_NAME
    encoded_jsonl_path = encoded_directory / ENCODED_JSONL_NAME
    if not encoded_jsonl_path.exists():
        raise FileNotFoundError(f"encoded JSONL does not exist: {encoded_jsonl_path}")

    snapshot = load_tokenizer_snapshot_json(tokenizer_snapshot_path)
    summary = extract_streaming_figure_artifacts(
        encoded_directory=encoded_directory,
        artifact_paths=artifact_paths,
        config=config,
        snapshot=snapshot,
        output_path=output_path,
        analysis_config_path=analysis_config_path,
        show_progress=show_progress,
        overwrite=overwrite,
        resume=resume,
    )

    return FigureExtractionResult(
        artifact_paths=artifact_paths,
        encoded_sample_count=summary.encoded_sample_count,
        profile_group_count=summary.profile_group_count,
        sample_profile_count=summary.sample_profile_count,
        extra_output_path=output_path,
    )


def copy_analysis_config(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == target_path.resolve():
        return

    shutil.copyfile(source_path, target_path)
