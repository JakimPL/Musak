from fractions import Fraction
from pathlib import Path

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.paths import RHYTHM_REFINER_CONFIG_PATH
from musak_model.rhythm_refiner import RhythmGridConfig, rhythm_grid_from_segment
from musak_model.rhythm_refiner.config import (
    RhythmRefinerDataConfig,
    RhythmRefinerMaskingConfig,
    RhythmRefinerModelConfig,
    RhythmRefinerTrainingConfig,
)
from musak_model.rhythm_refiner.dataset import (
    RhythmRefinerDataset,
    collate_rhythm_refiner_examples,
    rhythm_refiner_frames_from_samples,
)
from musak_model.rhythm_refiner.model import RhythmRefinerModel
from musak_model.rhythm_refiner.vocabulary import (
    COACTIVITY_TARGET_STATE_COUNT,
    RHYTHM_INPUT_UNKNOWN_ID,
    RHYTHM_TARGET_STATE_COUNT,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, Hand, HandToken, NoteToken, RestToken, ScaleType, Token
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


def test_rhythm_refiner_training_config_loads_default_yaml() -> None:
    config = RhythmRefinerTrainingConfig.load(RHYTHM_REFINER_CONFIG_PATH)

    assert config.grid.grid_denominator == 16
    assert config.model.hidden_size > 0
    assert config.optimization.batch_size > 0


def test_rhythm_refiner_dataset_masks_inputs_and_collates(duration_vocabulary: DurationVocabulary) -> None:
    frame = rhythm_grid_from_segment(
        _segment(_tokens(duration_vocabulary)),
        duration_vocabulary=duration_vocabulary,
        config=RhythmGridConfig(grid_denominator=4),
    )
    dataset = RhythmRefinerDataset(
        (frame,),
        masking=RhythmRefinerMaskingConfig(mask_probability=0.5, seed=1),
        model_config=_model_config(),
    )

    example = dataset[0]
    batch = collate_rhythm_refiner_examples([example])

    assert example.activity_loss_mask.any()
    assert (example.input_state_ids[example.activity_loss_mask] == RHYTHM_INPUT_UNKNOWN_ID).all()
    assert batch.input_state_ids.shape == (1, len(frame.cells), 2)
    assert batch.target_state_ids.shape == (1, len(frame.cells), 2)
    assert batch.coactivity_target_ids.shape == (1, len(frame.cells))


def test_rhythm_refiner_model_forward_shapes(duration_vocabulary: DurationVocabulary) -> None:
    model_config = _model_config()
    frame = rhythm_grid_from_segment(
        _segment(_tokens(duration_vocabulary)),
        duration_vocabulary=duration_vocabulary,
        config=RhythmGridConfig(grid_denominator=4),
    )
    dataset = RhythmRefinerDataset(
        (frame,),
        masking=RhythmRefinerMaskingConfig(mask_probability=0.5, seed=1),
        model_config=model_config,
    )
    batch = collate_rhythm_refiner_examples([dataset[0]])
    model = RhythmRefinerModel(model_config)

    logits = model(batch)

    assert logits.activity.shape == (1, len(frame.cells), 2, RHYTHM_TARGET_STATE_COUNT)
    assert logits.coactivity.shape == (1, len(frame.cells), COACTIVITY_TARGET_STATE_COUNT)


def test_rhythm_refiner_frame_building_skips_samples_that_do_not_fit_grid(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    frames = rhythm_refiner_frames_from_samples(
        [
            _encoded_sample(token_vocabulary, bar_durations=(Fraction(5, 16),)),
            _encoded_sample(token_vocabulary, bar_durations=(Fraction(1),)),
        ],
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        grid_config=RhythmGridConfig(grid_denominator=8),
        data_config=RhythmRefinerDataConfig(max_cells=None),
        show_progress=False,
    )

    assert len(frames) == 1
    assert frames[0].bar_durations == (Fraction(1),)


def _model_config() -> RhythmRefinerModelConfig:
    return RhythmRefinerModelConfig(
        hidden_size=16,
        transformer_layers=1,
        attention_heads=2,
        feedforward_size=32,
        dropout=0.0,
        max_bar_count=4,
        max_cells_per_bar=16,
        max_distance_cells=32,
        max_time_numerator=8,
        max_time_denominator=16,
    )


def _tokens(duration_vocabulary: DurationVocabulary) -> list[Token]:
    quarter_id = duration_vocabulary.require_duration_id(Fraction(1, 4))
    dotted_half_id = duration_vocabulary.require_duration_id(Fraction(3, 4))
    whole_id = duration_vocabulary.require_duration_id(Fraction(1))
    return [
        HandToken(hand=Hand.RIGHT),
        NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
        RestToken(duration_id=dotted_half_id),
        HandToken(hand=Hand.LEFT),
        NoteToken(degree=5, accidental=0, octave_offset=0, duration_id=whole_id),
        BarToken(),
    ]


def _segment(tokens: list[Token]) -> Segment:
    return Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("sample.mxl"),
            difficulty_level=None,
        ),
    )


def _encoded_sample(
    token_vocabulary: TokenVocabulary,
    *,
    bar_durations: tuple[Fraction, ...],
) -> EncodedExercise:
    return EncodedExercise(
        token_ids=token_vocabulary.encode([BarToken()]),
        bar_positions=[0],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            bar_durations=bar_durations,
            window_start_bar=0,
            source_file=Path(f"sample-{bar_durations[0]}.mxl"),
            difficulty_level=None,
        ),
    )
