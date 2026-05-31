from fractions import Fraction
from pathlib import Path

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.data.schema import SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise, IngestionSplit
from musak_model.training.stages.auxiliary_profiles import split_musical_auxiliary_profile_metrics


def _target_config() -> MusicalAuxiliaryTargetConfig:
    return MusicalAuxiliaryTargetConfig(
        note_density_bucket_boundaries=(0.5, 1.0),
        rhythmic_diversity_bucket_boundaries=(0.5,),
        voice_independence_bucket_boundaries=(0.5,),
        hand_span_bucket_boundaries=(1,),
    )


def _sample(token_vocabulary: TokenVocabulary, *, accidental: int) -> EncodedExercise:
    quarter_id = token_vocabulary.duration_vocabulary.fraction_to_id(Fraction(1, 4))
    return EncodedExercise(
        token_ids=token_vocabulary.encode(
            [
                HandToken(hand=Hand.RIGHT),
                NoteToken(degree=1, accidental=accidental, octave_offset=0, duration_id=quarter_id),
            ]
        ),
        bar_positions=[0, 0],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("piece.mxl"),
        ),
    )


def test_split_musical_auxiliary_profile_metrics_compare_train_and_validation(
    token_vocabulary: TokenVocabulary,
) -> None:
    metrics = split_musical_auxiliary_profile_metrics(
        IngestionSplit(
            train=[_sample(token_vocabulary, accidental=0)],
            validation=[_sample(token_vocabulary, accidental=1)],
            invalid_files=[],
        ),
        token_vocabulary=token_vocabulary,
        target_config=_target_config(),
    )

    assert metrics["model/split/musical_auxiliary/count/train_samples"] == 1.0
    assert metrics["model/split/musical_auxiliary/count/validation_samples"] == 1.0
    assert metrics["model/split/musical_auxiliary/count/train_bars"] == 1.0
    assert metrics["model/split/musical_auxiliary/count/validation_bars"] == 1.0
    assert metrics["model/split/musical_auxiliary/train/rate/uses_accidentals_bucket_0"] == 1.0
    assert metrics["model/split/musical_auxiliary/validation/rate/uses_accidentals_bucket_1"] == 1.0
    assert metrics["model/split/musical_auxiliary/mean/uses_accidentals_total_variation_distance"] == 1.0
    assert metrics["model/split/musical_auxiliary/mean/bar_uses_accidentals_total_variation_distance"] == 1.0
    assert metrics["model/split/musical_auxiliary/count/comparable_distributions"] == 12.0


def test_split_musical_auxiliary_profile_metrics_handles_empty_validation_split(
    token_vocabulary: TokenVocabulary,
) -> None:
    metrics = split_musical_auxiliary_profile_metrics(
        IngestionSplit(train=[_sample(token_vocabulary, accidental=0)], validation=[], invalid_files=[]),
        token_vocabulary=token_vocabulary,
        target_config=_target_config(),
    )

    assert metrics["model/split/musical_auxiliary/count/validation_samples"] == 0.0
    assert metrics["model/split/musical_auxiliary/count/comparable_distributions"] == 0.0
