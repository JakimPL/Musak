from fractions import Fraction

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.evaluation.generation.auxiliary_metrics import musical_auxiliary_bucket_metrics
from musak_model.evaluation.generation.schema import ConstraintReport, GenerationSample
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import EndToken, Hand, HandToken, NoteToken, ScaleType
from musak_model.training.config import GenerationEvaluationConfig


def _generation_config() -> GenerationEvaluationConfig:
    return GenerationEvaluationConfig(
        enabled=True,
        every_epochs=5,
        soft_sample_count=1,
        hard_sample_count=0,
        max_new_tokens=16,
        temperature=1.0,
        top_k=1,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        bar_count=1,
        minimum_duration_denominator=16,
        allow_dotted_durations=True,
        max_notes_per_hand=5,
        maximum_onset_span_semitones=12,
        maximum_pitch_gap_semitones=12,
        maximum_static_hand_span_degrees=5,
        harmonic_logit_bias_enabled=False,
        harmonic_logit_bias_alpha=0.20,
    )


def _target_config() -> MusicalAuxiliaryTargetConfig:
    return MusicalAuxiliaryTargetConfig(
        note_density_bucket_boundaries=(0.5, 1.0),
        rhythmic_diversity_bucket_boundaries=(0.5,),
        voice_independence_bucket_boundaries=(0.5,),
        hand_span_bucket_boundaries=(1,),
    )


def _sample(*, duration_id: int, decode_error: str | None = None) -> GenerationSample:
    return GenerationSample(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=1, octave_offset=0, duration_id=duration_id),
            EndToken(),
        ],
        reached_end=True,
        generated_token_count=3,
        constraint_error=None,
        constraint_report=ConstraintReport(
            failed=False,
            valid_token_fraction=1.0,
            first_failure_step=None,
            error=None,
        ),
        diagnostics=None,
        decode_error=decode_error,
        harmonic_plan_windows=None,
        completed_bars=0,
        target_bar_count=1,
    )


def test_musical_auxiliary_bucket_metrics_report_sample_and_bar_bucket_distribution(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))

    metrics = musical_auxiliary_bucket_metrics(
        samples=[
            _sample(duration_id=quarter_id),
            _sample(duration_id=quarter_id, decode_error="invalid"),
        ],
        config=_generation_config(),
        target_config=_target_config(),
        duration_vocabulary=duration_vocabulary,
    )

    assert metrics["generation/musical_auxiliary/count/samples"] == 1.0
    assert metrics["generation/musical_auxiliary/count/skipped_decode_errors"] == 1.0
    assert metrics["generation/musical_auxiliary/count/bars"] == 1.0
    assert metrics["generation/musical_auxiliary/mean/note_density_bucket_id"] == 0.0
    assert metrics["generation/musical_auxiliary/rate/note_density_bucket_0"] == 1.0
    assert metrics["generation/musical_auxiliary/rate/uses_accidentals_bucket_1"] == 1.0
    assert metrics["generation/musical_auxiliary/rate/bar_note_density_bucket_0"] == 1.0
