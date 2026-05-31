import pytest
from pydantic import ValidationError

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig


def test_musical_auxiliary_target_config_derives_class_counts_from_boundaries() -> None:
    config = MusicalAuxiliaryTargetConfig(
        note_density_bucket_boundaries=(0.5, 1.0),
        rhythmic_diversity_bucket_boundaries=(0.25, 0.75),
        voice_independence_bucket_boundaries=(0.2,),
        hand_span_bucket_boundaries=(4, 9, 12),
    )

    assert config.note_density_class_count == 3
    assert config.rhythmic_diversity_class_count == 3
    assert config.voice_independence_class_count == 2
    assert config.hand_span_class_count == 4
    assert config.uses_accidentals_class_count == 2
    assert config.dotted_duration_class_count == 2


def test_musical_auxiliary_target_config_rejects_unsorted_boundaries() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        MusicalAuxiliaryTargetConfig(
            note_density_bucket_boundaries=(0.5, 0.25),
            rhythmic_diversity_bucket_boundaries=(0.25, 0.75),
            voice_independence_bucket_boundaries=(0.2,),
            hand_span_bucket_boundaries=(4, 9, 12),
        )


def test_musical_auxiliary_target_config_rejects_non_positive_float_boundaries() -> None:
    with pytest.raises(ValidationError, match="positive"):
        MusicalAuxiliaryTargetConfig(
            note_density_bucket_boundaries=(0.0, 0.25),
            rhythmic_diversity_bucket_boundaries=(0.25, 0.75),
            voice_independence_bucket_boundaries=(0.2,),
            hand_span_bucket_boundaries=(4, 9, 12),
        )
