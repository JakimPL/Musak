import json
from fractions import Fraction
from pathlib import Path

from musak_model.conditioning.harmony.planner import HarmonicPlan, HarmonicPlanAlternative
from musak_model.conditioning.harmony.schema import HarmonicPlanWindow, HarmonicSlotRole
from musak_model.evaluation.generation.artifacts import write_generation_sample_artifacts
from musak_model.evaluation.generation.schema import (
    ConstraintReport,
    GenerationEvaluationResult,
    GenerationSample,
    GenerationSampleSuite,
)
from musak_model.harmony.schema import Chord, ChordQuality
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
    assert manifest_records[0]["harmonic_plan_summary"] is None
    assert manifest_records[1]["suite"] == "hard"
    assert manifest_records[1]["musicxml_path"] is None
    assert manifest_records[1]["decode_error"] == "join-with-previous token needs decoded notes"
    assert (tmp_path / "soft/sample_0000.tokens.txt").read_text(encoding="utf-8").startswith("R 1(1:4)")
    assert (tmp_path / "soft/sample_0000.musicxml").exists()
    assert not (tmp_path / "hard/sample_0000.musicxml").exists()


def test_generation_sample_artifacts_include_harmonic_plan_inspection_fields(tmp_path: Path) -> None:
    duration_vocabulary = DurationVocabulary(TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1))
    plan = _harmonic_plan()
    result = GenerationEvaluationResult(
        metrics={"generation/soft/count/samples": 1.0},
        sample_suites=(
            GenerationSampleSuite(
                name="soft",
                samples=[_sample(tokens=[EndToken()], harmonic_plan=plan)],
            ),
        ),
    )

    write_generation_sample_artifacts(
        result,
        output_directory=tmp_path,
        config=_generation_config(),
        duration_vocabulary=duration_vocabulary,
    )

    [manifest_record] = [
        json.loads(line) for line in (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_record["harmonic_plan"][0]["slot_role"] == "opening"
    assert manifest_record["harmonic_plan"][0]["distance_to_end"] == 1
    assert manifest_record["harmonic_plan"][0]["score_terms"] == {"role": 0.8}
    assert manifest_record["harmonic_plan_summary"] == {
        "alternative_count": 2,
        "distance_to_end": [1, 0],
        "final_harmonic_function": "tonic",
        "final_root_degree": 1,
        "longest_same_chord_run": 1,
        "score": 3.5,
        "slot_roles": ["opening", "cadence"],
        "top_alternative_scores": [3.5, 2.0],
        "unique_chord_count": 2,
        "window_count": 2,
    }
    assert manifest_record["harmonic_plan_alternatives"][0]["score"] == 3.5
    assert manifest_record["harmonic_plan_alternatives"][1]["windows"][0]["root_degree"] == 1


def _sample(
    *,
    tokens: list[Token],
    decode_error: str | None = None,
    harmonic_plan: HarmonicPlan | None = None,
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
        harmonic_plan_windows=None if harmonic_plan is None else harmonic_plan.windows,
        completed_bars=0,
        target_bar_count=1,
        harmonic_plan=harmonic_plan,
    )


def _harmonic_plan() -> HarmonicPlan:
    tonic = Chord(root_degree=1, root_accidental=0, quality=ChordQuality.MAJOR)
    dominant = Chord(root_degree=5, root_accidental=0, quality=ChordQuality.MAJOR)
    selected_windows = (
        HarmonicPlanWindow(
            start=Fraction(0),
            end=Fraction(1),
            chord=dominant,
            slot_role=HarmonicSlotRole.OPENING,
            distance_to_end=1,
            cadence_strength=0.15,
            tension_level=0.0,
            plan_confidence=1.0,
            score_terms={"role": 0.8},
        ),
        HarmonicPlanWindow(
            start=Fraction(1),
            end=Fraction(2),
            chord=tonic,
            slot_role=HarmonicSlotRole.CADENCE,
            distance_to_end=0,
            cadence_strength=1.0,
            tension_level=0.0,
            plan_confidence=1.0,
            score_terms={"terminal": 4.0},
        ),
    )
    fallback_windows = (
        HarmonicPlanWindow(
            start=Fraction(0),
            end=Fraction(1),
            chord=tonic,
            slot_role=HarmonicSlotRole.OPENING,
            distance_to_end=1,
        ),
        selected_windows[1],
    )
    return HarmonicPlan(
        windows=selected_windows,
        score=3.5,
        alternatives=(
            HarmonicPlanAlternative(windows=selected_windows, score=3.5),
            HarmonicPlanAlternative(windows=fallback_windows, score=2.0),
        ),
    )
