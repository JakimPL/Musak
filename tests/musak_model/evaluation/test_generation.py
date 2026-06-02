from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from torch import Tensor

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.conditioning.config import ConditioningConfig, DifficultyConfig, HarmonicConditioningConfig
from musak_model.conditioning.harmony.schema import HarmonicPlanInputTensors
from musak_model.conditioning.time_signature import TimeSignatureVocabularyConfig
from musak_model.evaluation.generation import GenerationSuiteEvaluator
from musak_model.model.config import (
    CNNConfig,
    GRUConfig,
    ModelConfig,
    ModelInputConfig,
    ModelOutputConfig,
    ModelOutputMode,
    TokenInputEmbeddingMode,
    TransformerConfig,
)
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.artifacts import figure_artifact_paths
from musak_model.n_grams.profile.builder import build_figure_profile, build_figure_sample_counts
from musak_model.n_grams.profile.loading import FigureProfileArtifacts
from musak_model.n_grams.profile.rhythm.io import build_rhythm_profile
from musak_model.n_grams.profile.rhythm.loading import RhythmProfileArtifacts
from musak_model.n_grams.profile.rhythm.schema import (
    RhythmCountKey,
    RhythmProfileMetadata,
    rhythm_artifact_paths_for_figure_root,
)
from musak_model.n_grams.profile.schema import FigureProfileMetadata
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Hand, HandToken, NoteToken, RestToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.config import GenerationEvaluationConfig, TrainingConditioningConfig


def _generation_config(**overrides: object) -> GenerationEvaluationConfig:
    values = {
        "enabled": True,
        "every_epochs": 5,
        "soft_sample_count": 1,
        "hard_sample_count": 1,
        "max_new_tokens": 16,
        "temperature": 1.0,
        "top_k": 1,
        "scale_root": 0,
        "scale_type": ScaleType.MAJOR,
        "time_numerator": 4,
        "time_denominator": 4,
        "bar_count": 1,
        "minimum_duration_denominator": 16,
        "allow_dotted_durations": True,
        "max_notes_per_hand": 5,
        "maximum_onset_span_semitones": 12,
        "maximum_pitch_gap_semitones": 12,
        "maximum_static_hand_span_degrees": 5,
    }
    values.update(overrides)
    return GenerationEvaluationConfig.model_validate(values)


def _conditioning_config(*, use_harmony_conditioning: bool = False) -> TrainingConditioningConfig:
    return TrainingConditioningConfig(
        use_time_signature=False,
        use_scale_type=False,
        use_difficulty=False,
        use_structural_conditioning=False,
        use_harmony_conditioning=use_harmony_conditioning,
        use_validity_penalty=False,
        validity_penalty_weight=0.05,
    )


def _musical_auxiliary_target_config() -> MusicalAuxiliaryTargetConfig:
    return MusicalAuxiliaryTargetConfig(
        note_density_bucket_boundaries=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
        rhythmic_diversity_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
        voice_independence_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
        hand_span_bucket_boundaries=(3, 5, 8, 12, 16),
    )


class ScriptedModel:
    def __init__(self, token_ids: list[int], *, vocabulary_size: int) -> None:
        self._token_ids = token_ids
        self._vocabulary_size = vocabulary_size
        self._step = 0
        self.training = True
        self.harmonic_plans: list[HarmonicPlanInputTensors | None] = []

    def eval(self) -> "ScriptedModel":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "ScriptedModel":
        self.training = mode
        return self

    def __call__(
        self,
        token_ids: Tensor,
        *,
        bar_positions: Tensor,
        bar_relative_ticks: Tensor,
        bar_duration_ticks: Tensor,
        active_hand_ids: Tensor,
        difficulty_ids: Tensor | None = None,
        scale_type_ids: Tensor | None = None,
        time_signature_ids: Tensor | None = None,
        structural_control_ids: Tensor | None = None,
        harmonic_plan: HarmonicPlanInputTensors | None = None,
        token_padding_mask: Tensor | None = None,
    ) -> Tensor:
        self.harmonic_plans.append(harmonic_plan)
        logits = torch.full((1, token_ids.size(1), self._vocabulary_size), -1000.0)
        scripted_id = self._token_ids[min(self._step, len(self._token_ids) - 1)]
        logits[0, -1, scripted_id] = 1000.0
        self._step += 1
        return logits


def test_generation_suite_logs_soft_and_hard_constraint_metrics() -> None:
    duration_vocabulary = DurationVocabulary(TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1))
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    whole_id = duration_vocabulary.fraction_to_id(Fraction(1, 1))
    scripted_ids = token_vocabulary.encode(
        [
            HandToken(hand=Hand.RIGHT),
            RestToken(duration_id=whole_id),
            HandToken(hand=Hand.LEFT),
            RestToken(duration_id=whole_id),
            BarToken(),
            EndToken(),
        ]
    )
    evaluator = GenerationSuiteEvaluator(
        config=_generation_config(),
        conditioning=_conditioning_config(),
        model_config=_model_config(token_vocabulary.vocabulary_size),
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        include_bar_count_control=False,
        figure_profile_artifacts=None,
    )

    result = evaluator.evaluate_result(
        ScriptedModel(scripted_ids, vocabulary_size=token_vocabulary.vocabulary_size),
        device=torch.device("cpu"),
    )
    metrics = result.metrics

    assert [suite.name for suite in result.sample_suites] == ["soft", "hard"]
    assert [len(suite.samples) for suite in result.sample_suites] == [1, 1]
    assert metrics["generation/soft/count/samples"] == 1.0
    assert metrics["generation/hard/count/samples"] == 1.0
    assert metrics["generation/soft/rate/end"] == 1.0
    assert metrics["generation/soft/rate/target_bar_completion"] == 1.0
    assert metrics["generation/soft/mean/bar_count_error"] == 0.0
    assert metrics["generation/soft/rate/constraint_failure"] == 0.0
    assert metrics["generation/soft/rate/empty_score"] == 1.0
    assert metrics["generation/soft/mean/accidental_note_fraction"] == 0.0
    assert metrics["generation/soft/mean/in_scale_note_fraction"] == 0.0
    assert metrics["generation/soft/mean/note_density_per_beat"] == 0.0
    assert metrics["generation/soft/rate/has_dotted_notes"] == 0.0
    assert metrics["generation/soft/mean/max_notes_per_onset"] == 0.0
    assert metrics["generation/hard/mean/constraint_valid_token_fraction"] == 1.0
    assert metrics["generation/musical_auxiliary/count/samples"] == 2.0
    assert "generation/soft/mean/sample_penalty" in metrics


def test_generation_suite_passes_harmonic_plan_when_enabled() -> None:
    duration_vocabulary = DurationVocabulary(TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1))
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    model = ScriptedModel([token_vocabulary.end_token_id], vocabulary_size=token_vocabulary.vocabulary_size)
    evaluator = GenerationSuiteEvaluator(
        config=_generation_config(soft_sample_count=1, hard_sample_count=0, max_new_tokens=1),
        conditioning=_conditioning_config(use_harmony_conditioning=True),
        model_config=_model_config(token_vocabulary.vocabulary_size, harmony_enabled=True),
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        include_bar_count_control=False,
        figure_profile_artifacts=None,
    )

    evaluator.evaluate_result(model, device=torch.device("cpu"))

    assert len(model.harmonic_plans) == 1
    assert model.harmonic_plans[0] is not None
    assert model.harmonic_plans[0].shape == torch.Size([1, 1])
    assert int(model.harmonic_plans[0].root_degree_ids[0, 0].item()) > 0


def test_generation_suite_rejects_harmony_conditioning_mismatch() -> None:
    duration_vocabulary = DurationVocabulary(TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1))
    token_vocabulary = TokenVocabulary(duration_vocabulary)

    with pytest.raises(ValueError, match="harmony conditioning"):
        GenerationSuiteEvaluator(
            config=_generation_config(soft_sample_count=0, hard_sample_count=0),
            conditioning=_conditioning_config(use_harmony_conditioning=True),
            model_config=_model_config(token_vocabulary.vocabulary_size, harmony_enabled=False),
            token_vocabulary=token_vocabulary,
            duration_vocabulary=duration_vocabulary,
            include_bar_count_control=False,
            figure_profile_artifacts=None,
        )


def test_generation_suite_includes_loaded_figure_profile_context_metrics(tmp_path: Path) -> None:
    duration_vocabulary = DurationVocabulary(TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1))
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    evaluator = GenerationSuiteEvaluator(
        config=_generation_config(soft_sample_count=0, hard_sample_count=0),
        conditioning=_conditioning_config(),
        model_config=_model_config(token_vocabulary.vocabulary_size),
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        include_bar_count_control=False,
        figure_profile_artifacts=_figure_profile_artifacts(tmp_path),
    )

    metrics = evaluator.evaluate(
        ScriptedModel([token_vocabulary.end_token_id], vocabulary_size=token_vocabulary.vocabulary_size),
        device=torch.device("cpu"),
    )

    assert metrics["generation/figure/count/profile_samples"] == 1.0
    assert metrics["generation/figure/count/profile_groups"] == 1.0
    assert metrics["generation/figure/count/sample_profiles"] == 1.0


def test_generation_suite_compares_generated_figures_to_loaded_profile(tmp_path: Path) -> None:
    duration_vocabulary = DurationVocabulary(TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1))
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    scripted_ids = token_vocabulary.encode(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            EndToken(),
        ]
    )
    evaluator = GenerationSuiteEvaluator(
        config=_generation_config(soft_sample_count=1, hard_sample_count=0),
        conditioning=_conditioning_config(),
        model_config=_model_config(token_vocabulary.vocabulary_size),
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        include_bar_count_control=False,
        figure_profile_artifacts=_figure_profile_artifacts(tmp_path),
    )

    metrics = evaluator.evaluate(
        ScriptedModel(scripted_ids, vocabulary_size=token_vocabulary.vocabulary_size),
        device=torch.device("cpu"),
    )

    assert metrics["generation/figure/count/generated_profile_samples"] == 1.0
    assert metrics["generation/figure/count/comparable_groups"] == 1.0
    assert metrics["generation/figure/mean/total_relative_abs_error"] == 0.0
    assert metrics["generation/figure/mean/monophonic_rate_abs_error"] == 0.0
    assert metrics["generation/figure/mean/chords_only_rate_abs_error"] == 0.0
    assert metrics["generation/figure/mean/in_scale_rate_abs_error"] == 0.0
    assert metrics["generation/figure/count/distribution_groups"] == 1.0
    assert metrics["generation/figure/mean/identity_total_variation_distance"] == 0.0


def test_generation_suite_identity_distribution_distance_detects_different_figures(tmp_path: Path) -> None:
    duration_vocabulary = DurationVocabulary(TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1))
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    scripted_ids = token_vocabulary.encode(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=1, octave_offset=0, duration_id=quarter_id),
            EndToken(),
        ]
    )
    evaluator = GenerationSuiteEvaluator(
        config=_generation_config(soft_sample_count=1, hard_sample_count=0),
        conditioning=_conditioning_config(),
        model_config=_model_config(token_vocabulary.vocabulary_size),
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        include_bar_count_control=False,
        figure_profile_artifacts=_figure_profile_artifacts(tmp_path),
    )

    metrics = evaluator.evaluate(
        ScriptedModel(scripted_ids, vocabulary_size=token_vocabulary.vocabulary_size),
        device=torch.device("cpu"),
    )

    assert metrics["generation/figure/mean/identity_total_variation_distance"] == 1.0


def test_generation_suite_compares_generated_rhythm_to_loaded_profile(tmp_path: Path) -> None:
    duration_vocabulary = DurationVocabulary(TokenizationConfig(shortest_duration=16, allowed_tuplets=(3,), max_dots=1))
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    scripted_ids = token_vocabulary.encode(
        [
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            EndToken(),
        ]
    )
    evaluator = GenerationSuiteEvaluator(
        config=_generation_config(soft_sample_count=1, hard_sample_count=0),
        conditioning=_conditioning_config(),
        model_config=_model_config(token_vocabulary.vocabulary_size),
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
        include_bar_count_control=False,
        figure_profile_artifacts=_figure_profile_artifacts(tmp_path, include_rhythm=True),
    )

    metrics = evaluator.evaluate(
        ScriptedModel(scripted_ids, vocabulary_size=token_vocabulary.vocabulary_size),
        device=torch.device("cpu"),
    )

    assert metrics["generation/rhythm/count/profile_samples"] == 1.0
    assert metrics["generation/rhythm/count/profile_groups"] == 1.0
    assert metrics["generation/rhythm/count/generated_profile_samples"] == 1.0
    assert metrics["generation/rhythm/count/duration_value_distribution_groups"] == 1.0
    assert metrics["generation/rhythm/mean/duration_value_total_variation_distance"] == 0.0


def test_generation_config_validates_minimum_duration_denominator() -> None:
    with pytest.raises(ValueError, match="power of two"):
        _generation_config(minimum_duration_denominator=12)


def _figure_profile_artifacts(tmp_path: Path, *, include_rhythm: bool = False) -> FigureProfileArtifacts:
    figure = FigureNGram(onsets=((((0, 0),), Fraction(1)),))
    profile = build_figure_profile(
        {
            ScaleType.MAJOR: {
                Hand.RIGHT: {
                    1: Counter({figure: 1}),
                }
            }
        },
        FigureProfileMetadata(min_n=1, max_n=1, sample_count=1),
    )
    sample_counts = build_figure_sample_counts(
        sample_index=0,
        scale_type=ScaleType.MAJOR,
        counts_by_hand={
            Hand.RIGHT: {
                1: Counter({figure: 1}),
            }
        },
    )
    return FigureProfileArtifacts(
        paths=figure_artifact_paths(tmp_path / "encoded"),
        profile=profile,
        counts_by_scale={
            ScaleType.MAJOR: {
                Hand.RIGHT: {
                    1: Counter({figure: 1}),
                }
            }
        },
        sample_counts=(sample_counts,),
        rhythm=_rhythm_profile_artifacts(tmp_path) if include_rhythm else None,
    )


def _rhythm_profile_artifacts(tmp_path: Path) -> RhythmProfileArtifacts:
    counts = Counter(
        {
            RhythmCountKey(
                scale_type=ScaleType.MAJOR.value,
                time_signature="4/4",
                hand=Hand.RIGHT.value,
                kind="duration_value",
                parameter="",
                value="1/4",
            ): 1
        }
    )
    return RhythmProfileArtifacts(
        paths=rhythm_artifact_paths_for_figure_root(figure_artifact_paths(tmp_path / "encoded").root_directory),
        profile=build_rhythm_profile(
            counts,
            metadata=RhythmProfileMetadata(
                rhythm_min_n=2,
                rhythm_max_n=4,
                grid_alignment_denominators=(1, 2, 4, 8, 16),
                strong_beat_offsets=(Fraction(0),),
                sample_count=1,
            ),
        ),
        counts=counts,
    )


def _model_config(vocabulary_size: int, *, harmony_enabled: bool = False) -> ModelConfig:
    return ModelConfig(
        vocabulary_size=vocabulary_size,
        duration_vocabulary_size=1,
        input=ModelInputConfig(embedding_mode=TokenInputEmbeddingMode.FLAT),
        output=ModelOutputConfig(mode=ModelOutputMode.FLAT),
        musical_auxiliary_targets=_musical_auxiliary_target_config(),
        cnn=CNNConfig(enabled=True, out_channels=16, kernel_sizes=(3,), num_layers=1, dropout=0.0),
        gru=GRUConfig(enabled=True, hidden_size=16, num_layers=1, dropout=0.0, bidirectional=False),
        transformer=TransformerConfig(
            hidden_size=16,
            num_heads=2,
            num_layers=1,
            feedforward_size=32,
            dropout=0.0,
            max_sequence_length=64,
        ),
        conditioning=ConditioningConfig(
            difficulty=DifficultyConfig(max_level=5),
            time_signature=TimeSignatureVocabularyConfig(max_denominator=4, relative_numerator_range=2),
            harmony=HarmonicConditioningConfig(enabled=harmony_enabled),
            cfg_dropout_probability=0.0,
        ),
    )
