from musak_model.auxiliary.config import BOOLEAN_TARGET_CLASS_COUNT, MusicalAuxiliaryTargetConfig
from musak_model.auxiliary.schema import (
    MUSICAL_AUXILIARY_TARGET_IGNORE_ID,
    MusicalAuxiliaryLogits,
    MusicalAuxiliaryTargetIds,
    MusicalAuxiliaryTargetTensors,
)
from musak_model.auxiliary.targets import (
    musical_auxiliary_target_ids_from_difficulty_features,
    musical_auxiliary_target_tensors_from_ids,
    stack_musical_auxiliary_targets,
)

__all__ = [
    "BOOLEAN_TARGET_CLASS_COUNT",
    "MUSICAL_AUXILIARY_TARGET_IGNORE_ID",
    "MusicalAuxiliaryLogits",
    "MusicalAuxiliaryTargetConfig",
    "MusicalAuxiliaryTargetIds",
    "MusicalAuxiliaryTargetTensors",
    "musical_auxiliary_target_ids_from_difficulty_features",
    "musical_auxiliary_target_tensors_from_ids",
    "stack_musical_auxiliary_targets",
]
