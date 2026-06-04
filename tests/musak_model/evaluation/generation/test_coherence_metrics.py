from dataclasses import dataclass
from fractions import Fraction

from musak_model.evaluation.generation.coherence_metrics import coherence_profile_metrics
from musak_model.evaluation.generation.schema import ConstraintReport, GenerationSample
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Hand, HandToken, NoteToken, RestToken, ScaleType, Token


@dataclass
class _Options:
    enabled: bool = True
    every_epochs: int = 1
    soft_sample_count: int = 1
    hard_sample_count: int = 1
    max_new_tokens: int = 16
    seed: int = 1729
    temperature: float = 1.0
    top_k: int | None = 1
    scale_root: int = 0
    scale_type: ScaleType = ScaleType.MAJOR
    time_numerator: int = 4
    time_denominator: int = 4
    bar_count: int = 1
    minimum_duration_denominator: int | None = 16
    allow_dotted_durations: bool = True
    max_notes_per_hand: int | None = 5
    maximum_onset_span_semitones: int | None = 12
    maximum_pitch_gap_semitones: int | None = 12
    maximum_static_hand_span_degrees: int | None = 5


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
        target_bar_count=1,
    )


def _note(degree: int, duration_id: int) -> NoteToken:
    return NoteToken(degree=degree, accidental=0, octave_offset=0, duration_id=duration_id)


def test_coherence_profile_metrics_use_suite_namespace(duration_vocabulary: DurationVocabulary) -> None:
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    samples = [
        _sample(
            [
                HandToken(hand=Hand.RIGHT),
                _note(1, whole),
                HandToken(hand=Hand.LEFT),
                _note(1, whole),
                BarToken(),
                EndToken(),
            ]
        )
    ]

    metrics = coherence_profile_metrics(
        "soft",
        samples,
        config=_Options(),
        duration_vocabulary=duration_vocabulary,
    )

    assert metrics["generation/soft/coherence/count/samples"] == 1.0
    assert metrics["generation/soft/coherence/count/note_events"] == 2.0
    assert metrics["generation/soft/coherence/count/whole_bar_note_events"] == 2.0
    assert metrics["generation/soft/coherence/count/whole_note_or_longer_events"] == 2.0
    assert metrics["generation/soft/coherence/rate/samples_with_whole_bar_note"] == 1.0
    assert metrics["generation/soft/coherence/rate/samples_with_whole_note_or_longer"] == 1.0
    assert metrics["generation/soft/coherence/rate/final_both_hands_active"] == 1.0


def test_coherence_profile_metrics_skip_decode_errors(duration_vocabulary: DurationVocabulary) -> None:
    whole = duration_vocabulary.require_duration_id(Fraction(1))
    broken = _sample(
        [HandToken(hand=Hand.RIGHT), RestToken(duration_id=whole), EndToken()],
        decode_error="boom",
    )
    valid = _sample(
        [
            HandToken(hand=Hand.RIGHT),
            _note(1, whole),
            BarToken(),
            EndToken(),
        ]
    )

    metrics = coherence_profile_metrics(
        "hard",
        [valid, broken],
        config=_Options(),
        duration_vocabulary=duration_vocabulary,
    )

    assert metrics["generation/hard/coherence/count/samples"] == 1.0
    assert metrics["generation/hard/coherence/count/note_events"] == 1.0
