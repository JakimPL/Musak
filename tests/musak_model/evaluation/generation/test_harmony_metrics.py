from dataclasses import dataclass
from fractions import Fraction

from musak_model.conditioning.harmony.schema import HarmonicPlanWindow
from musak_model.evaluation.generation.harmony_metrics import harmonic_plan_metrics
from musak_model.evaluation.generation.schema import ConstraintReport, GenerationSample
from musak_model.harmony.schema import Chord, ChordQuality
from musak_model.n_grams.config import RhythmAnalysisConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    JoinWithPreviousToken,
    NoteToken,
    ScaleType,
    Token,
)


@dataclass(frozen=True)
class _Options:
    scale_root: int = 0
    scale_type: ScaleType = ScaleType.MAJOR
    time_numerator: int = 4
    time_denominator: int = 4
    bar_count: int = 1


def test_harmonic_plan_metrics_report_plan_agreement_and_harmonic_quality(
    duration_vocabulary: DurationVocabulary,
) -> None:
    whole_id = duration_vocabulary.require_duration_id(Fraction(1))
    samples = [
        _sample(
            tokens=[
                HandToken(hand=Hand.RIGHT),
                _note(1, duration_id=whole_id),
                _note(3, duration_id=whole_id),
                JoinWithPreviousToken(),
                _note(5, duration_id=whole_id),
                JoinWithPreviousToken(),
                HandToken(hand=Hand.LEFT),
                _note(1, duration_id=whole_id),
                BarToken(),
                EndToken(),
            ],
            harmonic_plan_windows=_tonic_plan(),
        )
    ]

    metrics = harmonic_plan_metrics(
        "soft",
        samples,
        config=_Options(),
        duration_vocabulary=duration_vocabulary,
        rhythm_config=_rhythm_config(),
    )

    assert metrics["generation/soft/harmony/count/planned_samples"] == 1.0
    assert metrics["generation/soft/harmony/count/decoded_samples"] == 1.0
    assert metrics["generation/soft/harmony/count/planned_windows"] == 1.0
    assert metrics["generation/soft/harmony/count/decoded_windows"] == 1.0
    assert metrics["generation/soft/harmony/rate/harmonic_function_agreement"] == 1.0
    assert metrics["generation/soft/harmony/rate/root_degree_agreement"] == 1.0
    assert metrics["generation/soft/harmony/rate/duration_weighted_chord_tone_coverage"] == 1.0
    assert metrics["generation/soft/harmony/rate/strong_beat_chord_tone_coverage"] == 1.0
    assert metrics["generation/soft/harmony/count/coincident_onset_pairs"] == 3.0
    assert metrics["generation/soft/harmony/rate/coincident_onset_triadic_consonance"] == 1.0
    assert metrics["generation/soft/harmony/rate/coincident_onset_perfect_consonance"] == 2 / 3
    assert metrics["generation/soft/harmony/rate/final_slot_closure"] == 1.0


def test_harmonic_plan_metrics_detect_non_chord_tone_coverage(
    duration_vocabulary: DurationVocabulary,
) -> None:
    whole_id = duration_vocabulary.require_duration_id(Fraction(1))
    samples = [
        _sample(
            tokens=[
                HandToken(hand=Hand.RIGHT),
                _note(2, duration_id=whole_id),
                BarToken(),
                EndToken(),
            ],
            harmonic_plan_windows=_tonic_plan(),
        )
    ]

    metrics = harmonic_plan_metrics(
        "hard",
        samples,
        config=_Options(),
        duration_vocabulary=duration_vocabulary,
        rhythm_config=_rhythm_config(),
    )

    assert metrics["generation/hard/harmony/rate/duration_weighted_chord_tone_coverage"] == 0.0
    assert metrics["generation/hard/harmony/rate/strong_beat_chord_tone_coverage"] == 0.0
    assert metrics["generation/hard/harmony/rate/final_slot_closure"] == 0.0


def test_harmonic_plan_metrics_skip_unplanned_and_decode_error_samples(
    duration_vocabulary: DurationVocabulary,
) -> None:
    whole_id = duration_vocabulary.require_duration_id(Fraction(1))
    samples = [
        _sample(
            tokens=[HandToken(hand=Hand.RIGHT), _note(1, duration_id=whole_id), EndToken()],
            harmonic_plan_windows=None,
        ),
        _sample(
            tokens=[HandToken(hand=Hand.RIGHT), _note(1, duration_id=whole_id), EndToken()],
            harmonic_plan_windows=_tonic_plan(),
            decode_error="invalid",
        ),
    ]

    metrics = harmonic_plan_metrics(
        "soft",
        samples,
        config=_Options(),
        duration_vocabulary=duration_vocabulary,
        rhythm_config=_rhythm_config(),
    )

    assert metrics == {
        "generation/soft/harmony/count/planned_samples": 1.0,
        "generation/soft/harmony/count/decoded_samples": 0.0,
        "generation/soft/harmony/count/planned_windows": 1.0,
    }


def _sample(
    *,
    tokens: list[Token],
    harmonic_plan_windows: tuple[HarmonicPlanWindow, ...] | None,
    decode_error: str | None = None,
) -> GenerationSample:
    return GenerationSample(
        tokens=tokens,
        reached_end=bool(tokens and isinstance(tokens[-1], EndToken)),
        generated_token_count=len(tokens),
        constraint_error=None,
        constraint_report=ConstraintReport(failed=False, valid_token_fraction=1.0, first_failure_step=None, error=None),
        diagnostics=None,
        decode_error=decode_error,
        harmonic_plan_windows=harmonic_plan_windows,
        completed_bars=sum(isinstance(token, BarToken) for token in tokens),
        target_bar_count=1,
    )


def _note(degree: int, *, duration_id: int) -> NoteToken:
    return NoteToken(degree=degree, accidental=0, octave_offset=0, duration_id=duration_id)


def _tonic_plan() -> tuple[HarmonicPlanWindow, ...]:
    return (
        HarmonicPlanWindow(
            start=Fraction(0),
            end=Fraction(1),
            chord=Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR),
        ),
    )


def _rhythm_config() -> RhythmAnalysisConfig:
    return RhythmAnalysisConfig(
        min_n=1,
        max_n=1,
        grid_alignment_denominators=(1,),
        strong_beat_offsets=(Fraction(0),),
    )
