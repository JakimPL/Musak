import logging
from pathlib import Path
from time import perf_counter

from musak_model.analysis.n_grams.config import NGramAnalysisConfig
from musak_model.analysis.n_grams.figure.samples.counter import count_encoded_exercises_figure_ngrams
from musak_model.analysis.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.analysis.n_grams.profile.builder import build_figure_profile
from musak_model.analysis.n_grams.profile.loading import (
    FigureProfileArtifacts,
    load_processed_figure_profile_artifacts,
)
from musak_model.analysis.n_grams.profile.metrics import (
    figure_distribution_metrics,
    figure_profile_comparison_metrics,
)
from musak_model.analysis.n_grams.profile.schema import FigureProfile, FigureProfileMetadata
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.ingestion.schema import EncodedExercise, IngestionSplit

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
    analysis_config_path: Path | None = None,
    workers: int,
    show_progress: bool = False,
) -> dict[str, float]:
    config = (
        NGramAnalysisConfig.load() if analysis_config_path is None else NGramAnalysisConfig.load(analysis_config_path)
    )
    _LOGGER.info(
        "Computing train/validation figure metrics: train_samples=%s validation_samples=%s min_n=%s max_n=%s "
        "batch_size=%s workers=%s",
        len(split.train),
        len(split.validation),
        config.min_n,
        config.max_n,
        config.batch_size,
        max(1, workers),
    )
    train_counts = _split_counts(
        split.train,
        split_name="train",
        config=config,
        token_vocabulary=token_vocabulary,
        workers=workers,
        show_progress=show_progress,
    )
    validation_counts = _split_counts(
        split.validation,
        split_name="validation",
        config=config,
        token_vocabulary=token_vocabulary,
        workers=workers,
        show_progress=show_progress,
    )
    _LOGGER.info("Building train/validation figure profiles")
    train_profile = build_figure_profile(
        train_counts,
        FigureProfileMetadata(min_n=config.min_n, max_n=config.max_n, sample_count=len(split.train)),
    )
    validation_profile = build_figure_profile(
        validation_counts,
        FigureProfileMetadata(min_n=config.min_n, max_n=config.max_n, sample_count=len(split.validation)),
    )
    metrics = {
        **_split_figure_profile_count_metrics(
            train_profile=train_profile,
            validation_profile=validation_profile,
        ),
        **figure_profile_comparison_metrics(
            reference_profile=train_profile,
            comparison_profile=validation_profile,
            metric_prefix="model/split/figure",
            require_comparison_samples=True,
        ),
        **figure_distribution_metrics(
            reference_counts=train_counts,
            comparison_counts=validation_counts,
            metric_prefix="model/split/figure",
        ),
    }
    _LOGGER.info("Computed %s train/validation figure metric(s)", len(metrics))
    return metrics


def _split_figure_profile_count_metrics(
    *,
    train_profile: FigureProfile,
    validation_profile: FigureProfile,
) -> dict[str, float]:
    return {
        "model/split/figure/count/train_samples": float(train_profile.metadata.sample_count),
        "model/split/figure/count/validation_samples": float(validation_profile.metadata.sample_count),
        "model/split/figure/count/train_profile_groups": float(len(train_profile.groups)),
        "model/split/figure/count/validation_profile_groups": float(len(validation_profile.groups)),
    }


def _split_counts(
    samples: list[EncodedExercise],
    *,
    split_name: str,
    config: NGramAnalysisConfig,
    token_vocabulary: TokenVocabulary,
    workers: int,
    show_progress: bool,
) -> FigureNGramCountsByScale:
    task_count = (len(samples) + config.batch_size - 1) // config.batch_size
    _LOGGER.info(
        "Counting %s figure n-grams: samples=%s batches=%s min_n=%s max_n=%s workers=%s",
        split_name,
        len(samples),
        task_count,
        config.min_n,
        config.max_n,
        max(1, workers),
    )
    started_at = perf_counter()
    counts = count_encoded_exercises_figure_ngrams(
        samples,
        duration_vocabulary=token_vocabulary.duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=config.min_n,
        max_n=config.max_n,
        workers=max(1, workers),
        batch_size=config.batch_size,
        show_progress=show_progress,
        progress_description=f"Counting {split_name} figure n-gram batches",
    )
    _LOGGER.info(
        "Finished counting %s figure n-grams in %.1fs",
        split_name,
        perf_counter() - started_at,
    )
    return counts
