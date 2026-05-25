from pathlib import Path

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


def load_generation_figure_profile_artifacts(
    *,
    source_directory: Path,
    ingestion_config: IngestionConfig,
    tokenization_config: TokenizationConfig,
) -> FigureProfileArtifacts | None:
    if ingestion_config.processed_root is None:
        return None

    return load_processed_figure_profile_artifacts(
        processed_root=ingestion_config.processed_root,
        dataset_root=source_directory,
        tokenization_config=tokenization_config,
    )


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
    train_counts = _split_counts(
        split.train,
        config=config,
        token_vocabulary=token_vocabulary,
        workers=workers,
        show_progress=show_progress,
    )
    validation_counts = _split_counts(
        split.validation,
        config=config,
        token_vocabulary=token_vocabulary,
        workers=workers,
        show_progress=show_progress,
    )
    train_profile = build_figure_profile(
        train_counts,
        FigureProfileMetadata(min_n=config.min_n, max_n=config.max_n, sample_count=len(split.train)),
    )
    validation_profile = build_figure_profile(
        validation_counts,
        FigureProfileMetadata(min_n=config.min_n, max_n=config.max_n, sample_count=len(split.validation)),
    )
    return {
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
    config: NGramAnalysisConfig,
    token_vocabulary: TokenVocabulary,
    workers: int,
    show_progress: bool,
) -> FigureNGramCountsByScale:
    return count_encoded_exercises_figure_ngrams(
        samples,
        duration_vocabulary=token_vocabulary.duration_vocabulary,
        token_vocabulary=token_vocabulary,
        min_n=config.min_n,
        max_n=config.max_n,
        workers=max(1, workers),
        batch_size=config.batch_size,
        show_progress=show_progress,
    )
