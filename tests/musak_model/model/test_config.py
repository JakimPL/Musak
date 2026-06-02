from typing import Final

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.model.config import ModelConfig, ModelOutputMode, TokenInputEmbeddingMode
from musak_model.tokens.factorized import flat_vocabulary_attributes

DURATION_VOCABULARY_SIZE: Final[int] = 1


def test_model_config_loads_default_input_embedding_mode() -> None:
    config = ModelConfig.load(
        vocabulary_size=len(flat_vocabulary_attributes(duration_vocabulary_size=DURATION_VOCABULARY_SIZE)),
        duration_vocabulary_size=DURATION_VOCABULARY_SIZE,
        output_mode=ModelOutputMode.FACTORIZED,
        musical_auxiliary_targets=MusicalAuxiliaryTargetConfig(
            note_density_bucket_boundaries=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
            rhythmic_diversity_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
            voice_independence_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
            hand_span_bucket_boundaries=(3, 5, 8, 12, 16),
        ),
    )

    assert config.input.embedding_mode == TokenInputEmbeddingMode.FLAT_PLUS_FACTORIZED
