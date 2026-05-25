from pathlib import Path

from musak_model.analysis.n_grams.profile.loading import (
    FigureProfileArtifacts,
    load_processed_figure_profile_artifacts,
)
from musak_model.tokens.config import TokenizationConfig
from musak_model.training.ingestion.config import IngestionConfig


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
