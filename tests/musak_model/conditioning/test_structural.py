from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.conditioning.structural import (
    FALSE_CONTROL_ID,
    TRUE_CONTROL_ID,
    UNKNOWN_CONTROL_ID,
    FractionBucketConfig,
    IntegerBucketConfig,
    StructuralConditioningConfig,
    StructuralControlFeatures,
    StructuralControlVocabulary,
    extract_structural_control_features,
)
from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, JoinWithPreviousToken, NoteToken, RestToken, ScaleType


def _metadata() -> SegmentMetadata:
    return SegmentMetadata(
        key_root=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        bar_count=1,
        window_start_bar=0,
        source_file=Path("piece.mxl"),
        difficulty_level=None,
    )


def test_extract_structural_control_features_from_tokens(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    eighth_id = duration_vocabulary.fraction_to_id(Fraction(1, 8))
    dotted_quarter_id = duration_vocabulary.fraction_to_id(Fraction(3, 8))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=3, accidental=0, octave_offset=0, duration_id=quarter_id),
            JoinWithPreviousToken(),
            NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=eighth_id),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=dotted_quarter_id),
            RestToken(duration_id=eighth_id),
        ],
        metadata=_metadata(),
    )

    features = extract_structural_control_features(segment, duration_vocabulary=duration_vocabulary)

    assert features.shortest_note_duration == Fraction(1, 8)
    assert features.has_dotted_notes is True
    assert features.max_notes_per_onset == 2
    assert features.max_melodic_gap_semitones == 3
    assert features.static_hand_span_degrees == 5


def test_structural_control_vocabulary_maps_features_to_bucket_ids() -> None:
    vocabulary = StructuralControlVocabulary(
        StructuralConditioningConfig(
            shortest_note_duration=FractionBucketConfig(thresholds=("1/8", "1/4")),
            max_notes_per_onset=IntegerBucketConfig(thresholds=(1, 2)),
            max_melodic_gap_semitones=IntegerBucketConfig(thresholds=(2, 7)),
            static_hand_span_degrees=IntegerBucketConfig(thresholds=(3, 5)),
        )
    )
    features = StructuralControlFeatures(
        shortest_note_duration=Fraction(1, 8),
        has_dotted_notes=True,
        max_notes_per_onset=3,
        max_melodic_gap_semitones=7,
        static_hand_span_degrees=5,
    )

    assert vocabulary.features_to_ids(features) == (1, TRUE_CONTROL_ID, 3, 2, 2)
    assert vocabulary.vocabulary_sizes == (4, 3, 4, 4, 4)


def test_structural_control_vocabulary_maps_missing_features_to_unknown() -> None:
    vocabulary = StructuralControlVocabulary(StructuralConditioningConfig())

    assert vocabulary.features_to_ids(None) == (UNKNOWN_CONTROL_ID,) * 5


def test_structural_control_config_rejects_unsorted_thresholds() -> None:
    with pytest.raises(ValueError, match="sorted"):
        IntegerBucketConfig(thresholds=(2, 1))


def test_boolean_control_ids_are_stable() -> None:
    assert FALSE_CONTROL_ID == 1
    assert TRUE_CONTROL_ID == 2
