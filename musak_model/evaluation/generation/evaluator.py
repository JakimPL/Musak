from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Final

import torch
from torch import Tensor

from musak_model.conditioning.structural.schema import StructuralControlFeatures
from musak_model.conditioning.structural.vocabulary import StructuralControlVocabulary
from musak_model.conditioning.time_signature import TimeSignatureVocabulary
from musak_model.evaluation.diagnostics import SegmentDiagnostics, diagnose_segment
from musak_model.evaluation.generation.figure_metrics import figure_profile_metrics
from musak_model.evaluation.generation.musical_metrics import musical_profile_metrics
from musak_model.evaluation.generation.protocols import (
    GenerationConditioningOptions,
    GenerationEvaluationOptions,
    GenerationModel,
)
from musak_model.evaluation.generation.rhythm.metrics import rhythm_profile_metrics
from musak_model.evaluation.generation.sampling import (
    bar_positions,
    constraint_report,
    constraints_from_config,
    minimum_duration,
    sample_token_id,
    scale_type_to_id,
    segment_from_tokens,
)
from musak_model.evaluation.generation.schema import GenerationSample
from musak_model.evaluation.generation.suite_metrics import suite_metrics
from musak_model.generation.constraints import (
    GenerationConstraintError,
    GenerationConstraints,
    allowed_next_token_ids,
    mask_disallowed_logits,
)
from musak_model.model.config import ModelConfig
from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.profile.loading import FigureProfileArtifacts
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, Token
from musak_model.tokens.vocabulary import TokenVocabulary

_LOGGER = logging.getLogger(__name__)

_SOFT_SUITE_NAME: Final = "soft"
_HARD_SUITE_NAME: Final = "hard"


@dataclass(frozen=True)
class _SampleSuites:
    soft_samples: list[GenerationSample]
    hard_samples: list[GenerationSample]


@dataclass(frozen=True)
class _SampledTokenIds:
    token_ids: list[int]
    constraint_error: str | None


@dataclass(frozen=True)
class _DecodedSample:
    tokens: list[Token]
    diagnostics: SegmentDiagnostics | None
    decode_error: str | None


@dataclass(frozen=True)
class _ConstrainedLogits:
    logits: Tensor
    constraint_error: str | None


class GenerationSuiteEvaluator:
    def __init__(
        self,
        *,
        config: GenerationEvaluationOptions,
        conditioning: GenerationConditioningOptions,
        model_config: ModelConfig,
        token_vocabulary: TokenVocabulary,
        duration_vocabulary: DurationVocabulary,
        include_bar_count_control: bool,
        figure_profile_artifacts: FigureProfileArtifacts | None,
    ) -> None:
        self._config = config
        self._conditioning = conditioning
        self._model_config = model_config
        self._token_vocabulary = token_vocabulary
        self._duration_vocabulary = duration_vocabulary
        self._include_bar_count_control = include_bar_count_control
        self._figure_profile_artifacts = figure_profile_artifacts

    def evaluate(
        self,
        model: GenerationModel,
        *,
        device: torch.device,
    ) -> dict[str, float]:
        _LOGGER.info(
            "Starting generation evaluation: soft_samples=%s hard_samples=%s max_new_tokens=%s device=%s",
            self._config.soft_sample_count,
            self._config.hard_sample_count,
            self._config.max_new_tokens,
            device,
        )
        started_at = perf_counter()
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                suites = self._sample_suites(model, device=device)
        finally:
            model.train(was_training)

        metrics = self._evaluation_metrics(suites)
        _LOGGER.info(
            "Finished generation evaluation in %.1fs: metrics=%s",
            perf_counter() - started_at,
            len(metrics),
        )
        return metrics

    def _sample_suites(self, model: GenerationModel, *, device: torch.device) -> _SampleSuites:
        _LOGGER.info("Generating soft evaluation samples")
        soft_started_at = perf_counter()
        soft_samples = self._sample_suite(
            model,
            device=device,
            sample_count=self._config.soft_sample_count,
            hard_constraints=False,
            seed_offset=0,
        )
        _LOGGER.info("Generated soft evaluation samples in %.1fs", perf_counter() - soft_started_at)

        _LOGGER.info("Generating hard-constrained evaluation samples")
        hard_started_at = perf_counter()
        hard_samples = self._sample_suite(
            model,
            device=device,
            sample_count=self._config.hard_sample_count,
            hard_constraints=True,
            seed_offset=self._config.soft_sample_count,
        )
        _LOGGER.info("Generated hard-constrained evaluation samples in %.1fs", perf_counter() - hard_started_at)
        return _SampleSuites(soft_samples=soft_samples, hard_samples=hard_samples)

    def _evaluation_metrics(self, suites: _SampleSuites) -> dict[str, float]:
        _LOGGER.info("Computing generation evaluation metrics")
        samples = [*suites.soft_samples, *suites.hard_samples]
        metrics = {
            **suite_metrics(_SOFT_SUITE_NAME, suites.soft_samples),
            **suite_metrics(_HARD_SUITE_NAME, suites.hard_samples),
            **figure_profile_metrics(
                self._figure_profile_artifacts,
                samples=samples,
                config=self._config,
                duration_vocabulary=self._duration_vocabulary,
            ),
            **rhythm_profile_metrics(
                self._figure_profile_artifacts,
                samples=samples,
                config=self._config,
                rhythm_config=NGramAnalysisConfig.load().rhythm_analysis,
                duration_vocabulary=self._duration_vocabulary,
            ),
            **musical_profile_metrics(
                samples=samples,
                config=self._config,
                duration_vocabulary=self._duration_vocabulary,
            ),
        }
        return metrics

    def _sample_suite(
        self,
        model: GenerationModel,
        *,
        device: torch.device,
        sample_count: int,
        hard_constraints: bool,
        seed_offset: int,
    ) -> list[GenerationSample]:
        return [
            self._sample(
                model,
                device=device,
                hard_constraints=hard_constraints,
                seed=self._config.seed + seed_offset + sample_index,
            )
            for sample_index in range(sample_count)
        ]

    def _sample(
        self,
        model: GenerationModel,
        *,
        device: torch.device,
        hard_constraints: bool,
        seed: int,
    ) -> GenerationSample:
        constraints = constraints_from_config(self._config)
        sampled_token_ids = self._sample_token_ids(
            model,
            constraints=constraints,
            device=device,
            hard_constraints=hard_constraints,
            seed=seed,
        )
        decoded_sample = self._decode_sample(sampled_token_ids.token_ids)
        return self._generation_sample(
            decoded_sample,
            constraint_error=sampled_token_ids.constraint_error,
            constraints=constraints,
        )

    def _sample_token_ids(
        self,
        model: GenerationModel,
        *,
        constraints: GenerationConstraints,
        device: torch.device,
        hard_constraints: bool,
        seed: int,
    ) -> _SampledTokenIds:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        token_ids: list[int] = []
        constraint_error: str | None = None

        for _ in range(self._config.max_new_tokens):
            if self._prefix_exceeds_model_context(token_ids):
                break

            logits = self._next_token_logits(model, token_ids=token_ids, device=device)
            if hard_constraints:
                constrained_logits = self._constrained_logits(token_ids, logits=logits, constraints=constraints)
                if constrained_logits.constraint_error is not None:
                    constraint_error = constrained_logits.constraint_error
                    break

                logits = constrained_logits.logits

            next_token_id = sample_token_id(
                logits,
                temperature=self._config.temperature,
                top_k=self._config.top_k,
                generator=generator,
            )
            token_ids.append(next_token_id)
            if isinstance(self._token_vocabulary.id_to_token(next_token_id), EndToken):
                break

        return _SampledTokenIds(token_ids=token_ids, constraint_error=constraint_error)

    def _prefix_exceeds_model_context(self, token_ids: list[int]) -> bool:
        model_input_length = len([self._token_vocabulary.start_token_id, *token_ids])
        return model_input_length > self._model_config.transformer.max_sequence_length

    def _next_token_logits(
        self,
        model: GenerationModel,
        *,
        token_ids: list[int],
        device: torch.device,
    ) -> Tensor:
        model_input_ids = [self._token_vocabulary.start_token_id, *token_ids]
        return model(
            torch.tensor([model_input_ids], dtype=torch.long, device=device),
            bar_positions=torch.tensor(
                [bar_positions(token_ids, token_vocabulary=self._token_vocabulary)],
                dtype=torch.long,
                device=device,
            ),
            scale_type_ids=self._scale_type_tensor(device=device),
            time_signature_ids=self._time_signature_tensor(device=device),
            structural_control_ids=self._structural_control_tensor(device=device),
        )[0, -1]

    def _constrained_logits(
        self,
        token_ids: list[int],
        *,
        logits: Tensor,
        constraints: GenerationConstraints,
    ) -> _ConstrainedLogits:
        try:
            allowed_ids = allowed_next_token_ids(
                token_ids,
                constraints=constraints,
                token_vocabulary=self._token_vocabulary,
                duration_vocabulary=self._duration_vocabulary,
            )
            return _ConstrainedLogits(
                logits=mask_disallowed_logits(logits, allowed_token_ids=allowed_ids),
                constraint_error=None,
            )
        except GenerationConstraintError as exception:
            return _ConstrainedLogits(logits=logits, constraint_error=str(exception))

    def _decode_sample(self, token_ids: list[int]) -> _DecodedSample:
        tokens = self._token_vocabulary.decode(token_ids)
        segment = segment_from_tokens(tokens, config=self._config)
        diagnostics: SegmentDiagnostics | None = None
        decode_error: str | None = None

        try:
            diagnostics = diagnose_segment(segment, duration_vocabulary=self._duration_vocabulary)
        except ValueError as exception:
            decode_error = str(exception)

        return _DecodedSample(tokens=tokens, diagnostics=diagnostics, decode_error=decode_error)

    def _generation_sample(
        self,
        decoded_sample: _DecodedSample,
        *,
        constraint_error: str | None,
        constraints: GenerationConstraints,
    ) -> GenerationSample:
        return GenerationSample(
            tokens=decoded_sample.tokens,
            reached_end=bool(decoded_sample.tokens and isinstance(decoded_sample.tokens[-1], EndToken)),
            generated_token_count=len(decoded_sample.tokens),
            constraint_error=constraint_error,
            constraint_report=constraint_report(
                decoded_sample.tokens,
                constraints=constraints,
                duration_vocabulary=self._duration_vocabulary,
            ),
            diagnostics=decoded_sample.diagnostics,
            decode_error=decoded_sample.decode_error,
            completed_bars=sum(isinstance(token, BarToken) for token in decoded_sample.tokens),
            target_bar_count=self._config.bar_count,
        )

    def _scale_type_tensor(self, *, device: torch.device) -> Tensor | None:
        if not self._conditioning.use_scale_type:
            return None

        return torch.tensor([scale_type_to_id(self._config.scale_type)], dtype=torch.long, device=device)

    def _time_signature_tensor(self, *, device: torch.device) -> Tensor | None:
        if not self._conditioning.use_time_signature:
            return None

        vocabulary = TimeSignatureVocabulary(self._model_config.conditioning.time_signature)
        return torch.tensor(
            [vocabulary.time_signature_to_id((self._config.time_numerator, self._config.time_denominator))],
            dtype=torch.long,
            device=device,
        )

    def _structural_control_tensor(self, *, device: torch.device) -> Tensor | None:
        if not self._conditioning.use_structural_conditioning:
            return None

        features = StructuralControlFeatures(
            shortest_note_duration=minimum_duration(self._config),
            has_dotted_notes=None if self._config.allow_dotted_durations else False,
            max_notes_per_onset=None,
            max_notes_per_hand=self._config.max_notes_per_hand,
            max_onset_span_semitones=self._config.maximum_onset_span_semitones,
            max_melodic_gap_semitones=self._config.maximum_pitch_gap_semitones,
            static_hand_span_degrees=self._config.maximum_static_hand_span_degrees,
            bar_count=self._config.bar_count if self._include_bar_count_control else None,
        )
        vocabulary = StructuralControlVocabulary(self._model_config.conditioning.structural)
        return torch.tensor([vocabulary.features_to_ids(features)], dtype=torch.long, device=device)
