import logging
from pathlib import Path
from time import perf_counter

from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.profile.loading import FigureProfileArtifacts, load_processed_figure_profile_artifacts
from musak_model.n_grams.profile.metrics.profile_comparison import figure_profile_comparison_metrics
from musak_model.paths import DEFAULT_TRAINING_FIGURE_DIRECTORY, N_GRAM_ANALYSIS_CONFIG_PATH
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.schema import IngestionSplit
from musak_model.training.stages.figure_profiles.build import build_split_artifacts
from musak_model.training.stages.figure_profiles.cache import split_cache_key
from musak_model.training.stages.figure_profiles.metrics import (
    figure_distribution_metrics_from_csv,
    split_figure_profile_count_metrics,
)

_LOGGER = logging.getLogger(__name__)


def load_generation_figure_profile_artifacts(
    *,
    source_directory: Path,
    ingestion_config: IngestionConfig,
    tokenization_config: TokenizationConfig,
) -> FigureProfileArtifacts | None:
    if ingestion_config.processed_root is None:
        return None

    _LOGGER.info("Loading generation figure profile artifacts")
    started_at = perf_counter()
    artifacts = load_processed_figure_profile_artifacts(
        processed_root=ingestion_config.processed_root,
        dataset_root=source_directory,
        tokenization_config=tokenization_config,
    )
    if artifacts is None:
        _LOGGER.info("No generation figure profile artifacts found")
    else:
        _LOGGER.info(
            "Loaded generation figure profile artifacts in %.1fs: profile_groups=%s sample_profiles=%s",
            perf_counter() - started_at,
            len(artifacts.profile.groups),
            len(artifacts.sample_counts),
        )
    return artifacts


def split_figure_profile_metrics(
    split: IngestionSplit,
    *,
    token_vocabulary: TokenVocabulary,
    tokenization_config: TokenizationConfig,
    analysis_config_path: Path | None = None,
    artifact_root: Path = DEFAULT_TRAINING_FIGURE_DIRECTORY,
    workers: int,
    show_progress: bool = False,
) -> dict[str, float]:
    config_path = N_GRAM_ANALYSIS_CONFIG_PATH if analysis_config_path is None else analysis_config_path
    loaded_config = NGramAnalysisConfig.load(config_path)
    config = loaded_config.model_copy(
        update={"execution": loaded_config.execution.model_copy(update={"workers": max(1, workers)})}
    )
    split_key = split_cache_key(
        split,
        config=config,
        token_vocabulary=token_vocabulary,
        tokenization_config=tokenization_config,
    )
    split_directory = artifact_root / split_key
    _LOGGER.info(
        "Computing train/validation figure metrics: train_samples=%s validation_samples=%s min_n=%s max_n=%s "
        "batch_size=%s workers=%s artifact_dir=%s",
        len(split.train),
        len(split.validation),
        config.figure_analysis.min_n,
        config.figure_analysis.max_n,
        config.execution.batch_size,
        config.execution.workers,
        split_directory,
    )
    train_artifacts = build_split_artifacts(
        split.train,
        split_name="train",
        split_directory=split_directory,
        config=config,
        tokenization_config=tokenization_config,
        state_key=split_key,
        show_progress=show_progress,
    )
    validation_artifacts = build_split_artifacts(
        split.validation,
        split_name="validation",
        split_directory=split_directory,
        config=config,
        tokenization_config=tokenization_config,
        state_key=split_key,
        show_progress=show_progress,
    )
    metrics = {
        **split_figure_profile_count_metrics(
            train_profile=train_artifacts.profile,
            validation_profile=validation_artifacts.profile,
        ),
        **figure_profile_comparison_metrics(
            reference_profile=train_artifacts.profile,
            comparison_profile=validation_artifacts.profile,
            metric_prefix="model/split/figure",
            require_comparison_samples=True,
        ),
        **figure_distribution_metrics_from_csv(
            reference_path=train_artifacts.paths.counts_path,
            comparison_path=validation_artifacts.paths.counts_path,
            metric_prefix="model/split/figure",
        ),
    }
    _LOGGER.info("Computed %s train/validation figure metric(s)", len(metrics))
    return metrics
