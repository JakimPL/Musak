import json
from fractions import Fraction
from pathlib import Path

from musak_model.evaluation.generation.artifacts import write_generation_sample_artifacts
from musak_model.evaluation.generation.schema import (
    ConstraintReport,
    GenerationEvaluationResult,
    GenerationSample,
    GenerationSampleSuite,
)
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import EndToken, Hand, HandToken, NoteToken, ScaleType, Token
from musak_model.training.config import GenerationEvaluationConfig


def _generation_config() -> GenerationEvaluationConfig:
    return GenerationEvaluationConfig(
        enabled=True,
        every_epochs=1,
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
    )


def test_generation_sample_artifacts_include_manifest_token_text_and_musicxml(tmp_path: Path) -> None:
    duration_vocabulary = DurationVocabulary(TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1))
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    result = GenerationEvaluationResult(
        metrics={"generation/soft/count/samples": 1.0},
        sample_suites=(
            GenerationSampleSuite(
                name="soft",
                samples=[
                    _sample(
                        tokens=[
                            HandToken(hand=Hand.RIGHT),
                            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
                            HandToken(hand=Hand.LEFT),
                            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
                            EndToken(),
                        ],
                    )
                ],
            ),
            GenerationSampleSuite(
                name="hard",
                samples=[_sample(tokens=[EndToken()], decode_error="join-with-previous token needs decoded notes")],
            ),
        ),
    )

    write_generation_sample_artifacts(
        result,
        output_directory=tmp_path,
        config=_generation_config(),
        duration_vocabulary=duration_vocabulary,
    )

    manifest_lines = (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    manifest_records = [json.loads(line) for line in manifest_lines]
    assert manifest_records[0]["suite"] == "soft"
    assert manifest_records[0]["token_text_path"] == "soft/sample_0000.tokens.txt"
    assert manifest_records[0]["musicxml_path"] == "soft/sample_0000.musicxml"
    assert manifest_records[0]["constraint_report"]["failed"] is False
    assert manifest_records[1]["suite"] == "hard"
    assert manifest_records[1]["musicxml_path"] is None
    assert manifest_records[1]["decode_error"] == "join-with-previous token needs decoded notes"
    assert (tmp_path / "soft/sample_0000.tokens.txt").read_text(encoding="utf-8").startswith("R 1(1:4)")
    assert (tmp_path / "soft/sample_0000.musicxml").exists()
    assert not (tmp_path / "hard/sample_0000.musicxml").exists()


def _sample(
    *,
    tokens: list[Token],
    decode_error: str | None = None,
) -> GenerationSample:
    return GenerationSample(
        tokens=tokens,
        reached_end=True,
        generated_token_count=len(tokens),
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
