import logging
from pathlib import Path
from time import perf_counter

from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.profile.artifacts import FigureArtifactPaths
from musak_model.n_grams.profile.streaming.executor import process_missing_batches
from musak_model.n_grams.profile.streaming.export import export_figure_artifacts
from musak_model.n_grams.profile.streaming.schema import FigureStoreSummary
from musak_model.n_grams.profile.streaming.state import figure_state_key
from musak_model.n_grams.profile.streaming.store import (
    FigureWorkStore,
    clear_figure_work,
    complete_reference_artifacts_exist,
    existing_figure_summary,
    figure_reference_database_path,
)
from musak_model.processing.paths import ENCODED_JSONL_NAME
from musak_model.processing.snapshot import TokenizerSnapshot
from musak_model.tokens.config import TokenizationConfig

_LOGGER = logging.getLogger(__name__)


def extract_streaming_figure_artifacts(
    *,
    encoded_directory: Path,
    artifact_paths: FigureArtifactPaths,
    config: NGramAnalysisConfig,
    snapshot: TokenizerSnapshot,
    output_path: Path | None,
    show_progress: bool,
    overwrite: bool,
    resume: bool,
) -> FigureStoreSummary:
    store_path = figure_reference_database_path(artifact_paths)
    state_key = figure_state_key(config=config, snapshot=snapshot)
    if overwrite:
        _LOGGER.info("Clearing existing figure artifacts before extraction: %s", artifact_paths.root_directory)
        clear_figure_work(artifact_paths)

    if complete_reference_artifacts_exist(artifact_paths) and not overwrite:
        _LOGGER.info("Reusing complete figure/rhythm artifacts: %s", artifact_paths.root_directory)
        return existing_figure_summary(artifact_paths)

    _LOGGER.info("Opening figure work store: %s", store_path)
    started_at = perf_counter()
    with FigureWorkStore(store_path, state_key=state_key, resume=resume) as store:
        _LOGGER.info("Opened figure work store in %.1fs", perf_counter() - started_at)
        process_missing_batches(
            store,
            encoded_jsonl_path=encoded_directory / ENCODED_JSONL_NAME,
            tokenization_config=TokenizationConfig.model_validate(snapshot.tokenization_config),
            config=config,
            show_progress=show_progress,
        )
        _LOGGER.info("Exporting figure artifacts")
        started_at = perf_counter()
        summary = export_figure_artifacts(
            store,
            artifact_paths=artifact_paths,
            output_path=output_path,
            config=config,
            limit_per_group=config.limit_per_group,
        )
        _LOGGER.info("Exported figure artifacts in %.1fs", perf_counter() - started_at)

    _LOGGER.info("Retained durable figure reference database: %s", store_path)
    return summary
