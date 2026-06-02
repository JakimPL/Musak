from dataclasses import dataclass
from fractions import Fraction

from musak_model.evaluation.generation.musical_metrics import musical_profile_metrics
from musak_model.evaluation.generation.schema import ConstraintReport, GenerationSample
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    NoteToken,
    RestToken,
    ScaleType,
    Token,
)


@dataclass(frozen=True)
class _Options:
    scale_root: int = 0
    scale_type: ScaleType = ScaleType.MAJOR
    time_numerator: int = 4
    time_denominator: int = 4
    bar_count: int = 2


def _sample(tokens: list[Token], *, decode_error: str | None = None) -> GenerationSample:
    return GenerationSample(
        tokens=tokens,
        reached_end=bool(tokens and isinstance(tokens[-1], EndToken)),
        generated_token_count=len(tokens),
        constraint_error=None,
        constraint_report=ConstraintReport(failed=False, valid_token_fraction=1.0, first_failure_step=None, error=None),
        diagnostics=None,
        decode_error=decode_error,
        harmonic_plan_windows=None,
        completed_bars=sum(isinstance(token, BarToken) for token in tokens),
        target_bar_count=2,
    )


def _note(degree: int, duration_id: int) -> NoteToken:
    return NoteToken(degree=degree, accidental=0, octave_offset=0, duration_id=duration_id)


def _consonance_tokens(duration_vocabulary: DurationVocabulary) -> list[Token]:
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    return [
        HandToken(hand=Hand.RIGHT),
        _note(1, whole),
        HandToken(hand=Hand.LEFT),
        _note(1, whole),  # octave -> consonant
        BarToken(),
        HandToken(hand=Hand.RIGHT),
        _note(1, whole),
        HandToken(hand=Hand.LEFT),
        _note(2, whole),  # minor seventh -> dissonant
        BarToken(),
        EndToken(),
    ]


def test_musical_profile_metrics_use_generation_prefix(duration_vocabulary: DurationVocabulary) -> None:
    samples = [_sample(_consonance_tokens(duration_vocabulary))]

    metrics = musical_profile_metrics(samples=samples, config=_Options(), duration_vocabulary=duration_vocabulary)

    assert metrics["generation/musical/count/coincident_onset_pairs"] == 2.0
    assert metrics["generation/musical/rate/triadic_harmonic_consonance"] == 0.5
    assert metrics["generation/musical/rate/perfect_harmonic_consonance"] == 0.5


def test_musical_profile_metrics_skip_decode_errors(duration_vocabulary: DurationVocabulary) -> None:
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    broken = _sample(
        [HandToken(hand=Hand.RIGHT), RestToken(duration_id=whole), EndToken()],
        decode_error="boom",
    )
    samples = [_sample(_consonance_tokens(duration_vocabulary)), broken]

    metrics = musical_profile_metrics(samples=samples, config=_Options(), duration_vocabulary=duration_vocabulary)

    assert metrics["generation/musical/count/coincident_onset_pairs"] == 2.0
    assert metrics["generation/musical/rate/triadic_harmonic_consonance"] == 0.5
    assert metrics["generation/musical/rate/perfect_harmonic_consonance"] == 0.5
