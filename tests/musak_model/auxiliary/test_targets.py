from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.auxiliary.schema import MUSICAL_AUXILIARY_TARGET_IGNORE_ID
from musak_model.auxiliary.targets import musical_auxiliary_target_ids_from_difficulty_features
from musak_model.data.schema import DifficultyFeatures


def test_musical_auxiliary_targets_use_configured_boundaries() -> None:
    config = MusicalAuxiliaryTargetConfig(
        note_density_bucket_boundaries=(1.0, 2.0),
        rhythmic_diversity_bucket_boundaries=(0.25, 0.75),
        voice_independence_bucket_boundaries=(0.1, 0.9),
        hand_span_bucket_boundaries=(2, 10),
    )

    targets = musical_auxiliary_target_ids_from_difficulty_features(
        DifficultyFeatures(
            max_right_hand_span_semitones=4,
            max_left_hand_span_semitones=12,
            notes_per_beat=0.8,
            rhythmic_diversity=0.8,
            voice_independence=0.5,
            has_accidentals=True,
            has_dotted_notes=False,
        ),
        config=config,
    )

    assert targets.note_density_id == 0
    assert targets.rhythmic_diversity_id == 2
    assert targets.voice_independence_id == 1
    assert targets.uses_accidentals_id == 1
    assert targets.dotted_duration_id == 0
    assert targets.hand_span_id == 2


def test_musical_auxiliary_targets_ignore_missing_features() -> None:
    config = MusicalAuxiliaryTargetConfig(
        note_density_bucket_boundaries=(1.0,),
        rhythmic_diversity_bucket_boundaries=(0.5,),
        voice_independence_bucket_boundaries=(0.5,),
        hand_span_bucket_boundaries=(8,),
    )

    targets = musical_auxiliary_target_ids_from_difficulty_features(None, config=config)

    assert targets.note_density_id == MUSICAL_AUXILIARY_TARGET_IGNORE_ID
    assert targets.rhythmic_diversity_id == MUSICAL_AUXILIARY_TARGET_IGNORE_ID
    assert targets.voice_independence_id == MUSICAL_AUXILIARY_TARGET_IGNORE_ID
    assert targets.uses_accidentals_id == MUSICAL_AUXILIARY_TARGET_IGNORE_ID
    assert targets.dotted_duration_id == MUSICAL_AUXILIARY_TARGET_IGNORE_ID
    assert targets.hand_span_id == MUSICAL_AUXILIARY_TARGET_IGNORE_ID
