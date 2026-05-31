from fractions import Fraction
from pathlib import Path

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.auxiliary.schema import MUSICAL_AUXILIARY_TARGET_IGNORE_ID
from musak_model.auxiliary.targets import (
    bar_musical_auxiliary_target_ids_from_segment,
    musical_auxiliary_target_ids_from_difficulty_features,
)
from musak_model.data.schema import DifficultyFeatures, Segment, SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Hand, HandToken, NoteToken, ScaleType


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


def test_bar_musical_auxiliary_targets_are_derived_from_segment_tokens(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    dotted_quarter_id = duration_vocabulary.fraction_to_id(Fraction(3, 8))
    config = MusicalAuxiliaryTargetConfig(
        note_density_bucket_boundaries=(0.5, 1.0),
        rhythmic_diversity_bucket_boundaries=(0.5,),
        voice_independence_bucket_boundaries=(0.5,),
        hand_span_bucket_boundaries=(1,),
    )
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            BarToken(),
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=2, accidental=1, octave_offset=0, duration_id=dotted_quarter_id),
            BarToken(),
            EndToken(),
        ],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=2,
            window_start_bar=0,
            source_file=Path("piece.mxl"),
        ),
    )

    targets = bar_musical_auxiliary_target_ids_from_segment(
        segment,
        duration_vocabulary=duration_vocabulary,
        config=config,
    )

    assert len(targets) == 2
    assert targets[0].note_density_id == 1
    assert targets[0].rhythmic_diversity_id == 0
    assert targets[0].voice_independence_id == 0
    assert targets[0].uses_accidentals_id == 0
    assert targets[0].dotted_duration_id == 0
    assert targets[0].hand_span_id == 0
    assert targets[1].note_density_id == 0
    assert targets[1].uses_accidentals_id == 1
    assert targets[1].dotted_duration_id == 1
