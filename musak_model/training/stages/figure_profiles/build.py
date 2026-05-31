import logging
from pathlib import Path
from time import perf_counter

from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.profile.artifacts import figure_artifact_paths_from_root
from musak_model.n_grams.profile.io import read_figure_profile
from musak_model.n_grams.profile.streaming.executor import (
    process_missing_sample_batches,
)
from musak_model.n_grams.profile.streaming.export import export_figure_artifacts
from musak_model.n_grams.profile.streaming.store import (
    FigureWorkStore,
    complete_figure_artifacts_exist,
    figure_reference_database_path,
)
from musak_model.tokens.config import TokenizationConfig
from musak_model.training.ingestion.schema import EncodedExercise
from musak_model.training.stages.figure_profiles.schema import SplitFigureArtifacts

_LOGGER = logging.getLogger(__name__)


def build_split_artifacts(
    samples: list[EncodedExercise],
    *,
    split_name: str,
    split_directory: Path,
    config: NGramAnalysisConfig,
    tokenization_config: TokenizationConfig,
    state_key: str,
    show_progress: bool,
) -> SplitFigureArtifacts:
    paths = figure_artifact_paths_from_root(split_directory / split_name)
    task_count = (len(samples) + config.execution.batch_size - 1) // config.execution.batch_size
    if complete_figure_artifacts_exist(paths):
        _LOGGER.info("Reusing %s split figure artifacts: %s", split_name, paths.root_directory)
        return SplitFigureArtifacts(profile=read_figure_profile(paths.profile_path), paths=paths)

    _LOGGER.info(
        "Counting %s split figure n-grams: samples=%s batches=%s min_n=%s max_n=%s workers=%s artifact_dir=%s",
        split_name,
        len(samples),
        task_count,
        config.figure_analysis.min_n,
        config.figure_analysis.max_n,
        config.execution.workers,
        paths.root_directory,
    )
    started_at = perf_counter()
    store_path = figure_reference_database_path(paths)
    with FigureWorkStore(store_path, state_key=f"{state_key}:{split_name}", resume=True) as store:
        process_missing_sample_batches(
            store,
            samples=samples,
            tokenization_config=tokenization_config,
            config=config,
            show_progress=show_progress,
            progress_description=f"Counting {split_name} split figure n-gram batches",
        )
        export_figure_artifacts(
            store,
            artifact_paths=paths,
            output_path=None,
            config=config,
            limit_per_group=None,
        )

    store_path.unlink(missing_ok=True)
    _LOGGER.info(
        "Saved %s split figure artifacts in %.1fs: counts=%s profile=%s",
        split_name,
        perf_counter() - started_at,
        paths.counts_path,
        paths.profile_path,
    )
    return SplitFigureArtifacts(profile=read_figure_profile(paths.profile_path), paths=paths)
