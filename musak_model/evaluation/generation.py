from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from musak_model.conditioning.structural.schema import StructuralControlFeatures
from musak_model.conditioning.structural.vocabulary import StructuralControlVocabulary
from musak_model.conditioning.time_signature import TimeSignatureVocabulary
from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.evaluation.diagnostics import SegmentDiagnostics, diagnose_segment
from musak_model.generation.constraints import (
    GenerationConstraintError,
    GenerationConstraints,
    allowed_next_token_ids,
    mask_disallowed_logits,
    state_from_tokens,
)
from musak_model.model.config import ModelConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken, ScaleType, Token
from musak_model.tokens.vocabulary import TokenVocabulary


class GenerationEvaluationOptions(Protocol):
    enabled: bool
    every_epochs: int
    soft_sample_count: int
    hard_sample_count: int
    max_new_tokens: int
    seed: int
    temperature: float
    top_k: int | None
    scale_root: int
    scale_type: ScaleType
    time_numerator: int
    time_denominator: int
    bar_count: int
    minimum_duration_denominator: int | None
    allow_dotted_durations: bool
    max_notes_per_hand: int | None
    maximum_onset_span_semitones: int | None
    maximum_pitch_gap_semitones: int | None
    maximum_static_hand_span_degrees: int | None


class GenerationConditioningOptions(Protocol):
    use_time_signature: bool
    use_scale_type: bool
    use_structural_conditioning: bool


_SOFT_SUITE_NAME = "soft"
_HARD_SUITE_NAME = "hard"


class GenerationModel(Protocol):
    def eval(self) -> GenerationModel: ...

    def train(self, mode: bool = True) -> GenerationModel: ...

    @property
    def training(self) -> bool: ...

    def __call__(
        self,
        token_ids: Tensor,
        *,
        bar_positions: Tensor,
        difficulty_ids: Tensor | None = None,
        scale_type_ids: Tensor | None = None,
        time_signature_ids: Tensor | None = None,
        structural_control_ids: Tensor | None = None,
        token_padding_mask: Tensor | None = None,
    ) -> Tensor: ...


@dataclass(frozen=True)
class GenerationSample:
    tokens: list[Token]
    reached_end: bool
    generated_token_count: int
    constraint_error: str | None
    constraint_report: ConstraintReport
    diagnostics: SegmentDiagnostics | None
    decode_error: str | None
    completed_bars: int
    target_bar_count: int


@dataclass(frozen=True)
class ConstraintReport:
    failed: bool
    valid_token_fraction: float
    first_failure_step: int | None
    error: str | None


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
    ) -> None:
        self._config = config
        self._conditioning = conditioning
        self._model_config = model_config
        self._token_vocabulary = token_vocabulary
        self._duration_vocabulary = duration_vocabulary
        self._include_bar_count_control = include_bar_count_control

    def evaluate(self, model: GenerationModel, *, device: torch.device) -> dict[str, float]:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                soft_samples = self._sample_suite(
                    model,
                    device=device,
                    sample_count=self._config.soft_sample_count,
                    hard_constraints=False,
                    seed_offset=0,
                )
                hard_samples = self._sample_suite(
                    model,
                    device=device,
                    sample_count=self._config.hard_sample_count,
                    hard_constraints=True,
                    seed_offset=self._config.soft_sample_count,
                )
        finally:
            model.train(was_training)

        return {
            **_suite_metrics(_SOFT_SUITE_NAME, soft_samples),
            **_suite_metrics(_HARD_SUITE_NAME, hard_samples),
        }

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
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        constraints = _constraints_from_config(self._config)
        token_ids: list[int] = []
        constraint_error: str | None = None

        for _ in range(self._config.max_new_tokens):
            model_input_ids = [self._token_vocabulary.start_token_id, *token_ids]
            if len(model_input_ids) > self._model_config.transformer.max_sequence_length:
                break

            logits = model(
                torch.tensor([model_input_ids], dtype=torch.long, device=device),
                bar_positions=torch.tensor(
                    [_bar_positions(token_ids, token_vocabulary=self._token_vocabulary)],
                    dtype=torch.long,
                    device=device,
                ),
                scale_type_ids=self._scale_type_tensor(device=device),
                time_signature_ids=self._time_signature_tensor(device=device),
                structural_control_ids=self._structural_control_tensor(device=device),
            )[0, -1]
            if hard_constraints:
                try:
                    allowed_ids = allowed_next_token_ids(
                        token_ids,
                        constraints=constraints,
                        token_vocabulary=self._token_vocabulary,
                        duration_vocabulary=self._duration_vocabulary,
                    )
                    logits = mask_disallowed_logits(logits, allowed_token_ids=allowed_ids)
                except GenerationConstraintError as exception:
                    constraint_error = str(exception)
                    break

            next_token_id = _sample_token_id(
                logits,
                temperature=self._config.temperature,
                top_k=self._config.top_k,
                generator=generator,
            )
            token_ids.append(next_token_id)
            if isinstance(self._token_vocabulary.id_to_token(next_token_id), EndToken):
                break

        tokens = self._token_vocabulary.decode(token_ids)
        segment = _segment_from_tokens(tokens, config=self._config)
        diagnostics: SegmentDiagnostics | None = None
        decode_error: str | None = None
        try:
            diagnostics = diagnose_segment(segment, duration_vocabulary=self._duration_vocabulary)
        except ValueError as exception:
            decode_error = str(exception)

        return GenerationSample(
            tokens=tokens,
            reached_end=bool(tokens and isinstance(tokens[-1], EndToken)),
            generated_token_count=len(tokens),
            constraint_error=constraint_error,
            constraint_report=_constraint_report(
                tokens,
                constraints=constraints,
                duration_vocabulary=self._duration_vocabulary,
            ),
            diagnostics=diagnostics,
            decode_error=decode_error,
            completed_bars=sum(isinstance(token, BarToken) for token in tokens),
            target_bar_count=self._config.bar_count,
        )

    def _scale_type_tensor(self, *, device: torch.device) -> Tensor | None:
        if not self._conditioning.use_scale_type:
            return None

        return torch.tensor([_scale_type_to_id(self._config.scale_type)], dtype=torch.long, device=device)

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
            shortest_note_duration=_minimum_duration(self._config),
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


def _suite_metrics(suite_name: str, samples: list[GenerationSample]) -> dict[str, float]:
    prefix = f"generation/{suite_name}"
    if not samples:
        return {f"{prefix}/count/samples": 0.0}

    diagnostics = [sample.diagnostics for sample in samples if sample.diagnostics is not None]
    metrics = {
        f"{prefix}/count/samples": float(len(samples)),
        f"{prefix}/rate/end": _rate(samples, lambda sample: sample.reached_end),
        f"{prefix}/rate/decode_error": _rate(samples, lambda sample: sample.decode_error is not None),
        f"{prefix}/rate/constraint_error": _rate(samples, lambda sample: sample.constraint_error is not None),
        f"{prefix}/rate/constraint_failure": _rate(samples, lambda sample: sample.constraint_report.failed),
        f"{prefix}/mean/constraint_valid_token_fraction": _mean(
            sample.constraint_report.valid_token_fraction for sample in samples
        ),
        f"{prefix}/mean/constraint_first_failure_step": _mean(
            float(sample.constraint_report.first_failure_step)
            for sample in samples
            if sample.constraint_report.first_failure_step is not None
        ),
        f"{prefix}/rate/target_bar_completion": _rate(
            samples,
            lambda sample: sample.reached_end and sample.completed_bars == sample.target_bar_count,
        ),
        f"{prefix}/mean/bar_count_error": _mean(
            abs(sample.completed_bars - sample.target_bar_count) for sample in samples
        ),
        f"{prefix}/mean/generated_tokens": _mean(sample.generated_token_count for sample in samples),
        f"{prefix}/mean/completed_bars": _mean(sample.completed_bars for sample in samples),
        f"{prefix}/rate/empty_score": _rate(diagnostics, lambda item: item.empty_score),
        f"{prefix}/rate/one_hand_only": _rate(diagnostics, lambda item: item.one_hand_only),
        f"{prefix}/mean/right_silence_fraction": _mean(item.right_silence_fraction for item in diagnostics),
        f"{prefix}/mean/left_silence_fraction": _mean(item.left_silence_fraction for item in diagnostics),
        f"{prefix}/mean/both_hands_silence_fraction": _mean(item.both_hands_silence_fraction for item in diagnostics),
        f"{prefix}/mean/both_hands_active_fraction": _mean(item.both_hands_active_fraction for item in diagnostics),
        f"{prefix}/mean/hand_activity_balance": _mean(item.hand_activity_balance for item in diagnostics),
        f"{prefix}/mean/silent_bar_count": _mean(item.silent_bar_count for item in diagnostics),
        f"{prefix}/mean/silent_bar_fraction": _mean(item.silent_bar_fraction for item in diagnostics),
        f"{prefix}/mean/silent_edge_bar_count": _mean(item.silent_edge_bar_count for item in diagnostics),
        f"{prefix}/mean/note_token_fraction": _mean(item.note_token_fraction for item in diagnostics),
        f"{prefix}/mean/rest_token_fraction": _mean(item.rest_token_fraction for item in diagnostics),
        f"{prefix}/mean/hold_token_fraction": _mean(item.hold_token_fraction for item in diagnostics),
        f"{prefix}/mean/accidental_note_fraction": _mean(item.accidental_note_fraction for item in diagnostics),
        f"{prefix}/mean/in_scale_note_fraction": _mean(item.in_scale_note_fraction for item in diagnostics),
        f"{prefix}/mean/note_density_per_beat": _mean(item.note_density_per_beat for item in diagnostics),
        f"{prefix}/mean/onset_density_per_beat": _mean(item.onset_density_per_beat for item in diagnostics),
        f"{prefix}/mean/right_onset_density_per_beat": _mean(item.right_onset_density_per_beat for item in diagnostics),
        f"{prefix}/mean/left_onset_density_per_beat": _mean(item.left_onset_density_per_beat for item in diagnostics),
        f"{prefix}/mean/shortest_note_duration_beats": _mean(item.shortest_note_duration_beats for item in diagnostics),
        f"{prefix}/rate/has_dotted_notes": _rate(diagnostics, lambda item: item.has_dotted_notes),
        f"{prefix}/mean/max_notes_per_onset": _mean(item.max_notes_per_onset for item in diagnostics),
        f"{prefix}/mean/max_notes_per_hand": _mean(item.max_notes_per_hand for item in diagnostics),
        f"{prefix}/mean/max_onset_span_semitones": _mean(item.max_onset_span_semitones for item in diagnostics),
        f"{prefix}/mean/max_melodic_gap_semitones": _mean(item.max_melodic_gap_semitones for item in diagnostics),
        f"{prefix}/mean/static_hand_span_degrees": _mean(item.static_hand_span_degrees for item in diagnostics),
        f"{prefix}/mean/synchronized_onset_fraction": _mean(item.synchronized_onset_fraction for item in diagnostics),
        f"{prefix}/mean/independent_onset_fraction": _mean(item.independent_onset_fraction for item in diagnostics),
    }
    return {name: value for name, value in metrics.items() if math.isfinite(value)}


def _constraint_report(
    tokens: list[Token],
    *,
    constraints: GenerationConstraints,
    duration_vocabulary: DurationVocabulary,
) -> ConstraintReport:
    valid_count = 0
    prefix: list[Token] = []
    for step, token in enumerate(tokens, start=1):
        try:
            state_from_tokens(
                [*prefix, token],
                constraints=constraints,
                duration_vocabulary=duration_vocabulary,
            )
        except GenerationConstraintError as exception:
            return ConstraintReport(
                failed=True,
                valid_token_fraction=valid_count / max(len(tokens), 1),
                first_failure_step=step,
                error=str(exception),
            )

        prefix.append(token)
        valid_count += 1

    return ConstraintReport(
        failed=False,
        valid_token_fraction=valid_count / max(len(tokens), 1),
        first_failure_step=None,
        error=None,
    )


def _constraints_from_config(config: GenerationEvaluationOptions) -> GenerationConstraints:
    return GenerationConstraints(
        time_numerator=config.time_numerator,
        time_denominator=config.time_denominator,
        bar_count=config.bar_count,
        minimum_duration=_minimum_duration(config),
        allow_dotted_durations=config.allow_dotted_durations,
        max_notes_per_hand=config.max_notes_per_hand,
        maximum_onset_span_semitones=config.maximum_onset_span_semitones,
        maximum_pitch_gap_semitones=config.maximum_pitch_gap_semitones,
        maximum_static_hand_span_degrees=config.maximum_static_hand_span_degrees,
        scale_root=config.scale_root,
        scale_type=config.scale_type,
    )


def _segment_from_tokens(tokens: list[Token], *, config: GenerationEvaluationOptions) -> Segment:
    return Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            scale_root=config.scale_root,
            scale_type=config.scale_type,
            time_numerator=config.time_numerator,
            time_denominator=config.time_denominator,
            bar_count=config.bar_count,
            window_start_bar=0,
            source_file=Path("generation-evaluation"),
            difficulty_level=None,
        ),
    )


def _minimum_duration(config: GenerationEvaluationOptions) -> Fraction | None:
    if config.minimum_duration_denominator is None:
        return None

    return Fraction(1, config.minimum_duration_denominator)


def _scale_type_to_id(scale_type: ScaleType) -> int:
    return tuple(ScaleType).index(scale_type)


def _bar_positions(token_ids: list[int], *, token_vocabulary: TokenVocabulary) -> list[int]:
    positions = [0]
    bar_index = 0
    for token in token_vocabulary.decode(token_ids):
        positions.append(bar_index)
        if isinstance(token, BarToken):
            bar_index += 1

    return positions


def _sample_token_id(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator,
) -> int:
    filtered_logits = logits.clone()
    if top_k is not None and top_k < filtered_logits.numel():
        threshold = torch.topk(filtered_logits, top_k).values[-1]
        filtered_logits = torch.where(
            filtered_logits < threshold,
            torch.full_like(filtered_logits, float("-inf")),
            filtered_logits,
        )

    probabilities = torch.softmax(filtered_logits / temperature, dim=-1)
    if not torch.all(torch.isfinite(probabilities)):
        return int(torch.argmax(logits).item())

    return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())


def _rate[T](items: list[T], predicate: Callable[[T], bool]) -> float:
    if not items:
        return math.nan

    return sum(bool(predicate(item)) for item in items) / len(items)


def _mean(values: Iterable[float | int]) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return math.nan

    return sum(collected) / len(collected)
