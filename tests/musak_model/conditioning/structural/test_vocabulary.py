from fractions import Fraction

from musak_model.conditioning.structural.config import (
    DurationDenominatorBucketConfig,
    IntegerBucketConfig,
    StructuralConditioningConfig,
)
from musak_model.conditioning.structural.constants import TRUE_CONTROL_ID, UNKNOWN_CONTROL_ID
from musak_model.conditioning.structural.schema import StructuralControlFeatures
from musak_model.conditioning.structural.vocabulary import StructuralControlVocabulary


def test_structural_control_vocabulary_maps_features_to_bucket_ids() -> None:
    vocabulary = StructuralControlVocabulary(
        StructuralConditioningConfig(
            shortest_note_duration=DurationDenominatorBucketConfig(thresholds=(8, 4)),
            max_notes_per_onset=IntegerBucketConfig(thresholds=(1, 2)),
            max_notes_per_hand=IntegerBucketConfig(thresholds=(1, 2, 5)),
            max_onset_span_semitones=IntegerBucketConfig(thresholds=(4, 12)),
            max_melodic_gap_semitones=IntegerBucketConfig(thresholds=(2, 7)),
            static_hand_span_degrees=IntegerBucketConfig(thresholds=(3, 5)),
            bar_count=IntegerBucketConfig(thresholds=(1, 2, 4)),
        )
    )
    features = StructuralControlFeatures(
        shortest_note_duration=Fraction(1, 8),
        has_dotted_notes=True,
        max_notes_per_onset=3,
        max_notes_per_hand=5,
        max_onset_span_semitones=12,
        max_melodic_gap_semitones=7,
        static_hand_span_degrees=5,
        bar_count=4,
    )

    assert vocabulary.features_to_ids(features) == (1, TRUE_CONTROL_ID, 3, 3, 2, 2, 2, 3)
    assert vocabulary.vocabulary_sizes == (4, 3, 4, 5, 4, 4, 4, 5)


def test_structural_control_vocabulary_maps_missing_features_to_unknown() -> None:
    vocabulary = StructuralControlVocabulary(StructuralConditioningConfig())

    assert vocabulary.features_to_ids(None) == (UNKNOWN_CONTROL_ID,) * 8
