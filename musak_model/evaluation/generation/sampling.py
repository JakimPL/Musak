from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import torch
from torch import Tensor

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.evaluation.generation.protocols import GenerationEvaluationOptions
from musak_model.evaluation.generation.schema import ConstraintReport
from musak_model.generation.constraints import (
    GenerationConstraintError,
    GenerationConstraints,
    state_from_tokens,
)
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, ScaleType, Token
from musak_model.tokens.vocabulary import TokenVocabulary


def constraints_from_config(config: GenerationEvaluationOptions) -> GenerationConstraints:
    return GenerationConstraints(
        time_numerator=config.time_numerator,
        time_denominator=config.time_denominator,
        bar_count=config.bar_count,
        minimum_duration=minimum_duration(config),
        allow_dotted_durations=config.allow_dotted_durations,
        max_notes_per_hand=config.max_notes_per_hand,
        maximum_onset_span_semitones=config.maximum_onset_span_semitones,
        maximum_pitch_gap_semitones=config.maximum_pitch_gap_semitones,
        maximum_static_hand_span_degrees=config.maximum_static_hand_span_degrees,
        scale_root=config.scale_root,
        scale_type=config.scale_type,
    )


def segment_from_tokens(tokens: list[Token], *, config: GenerationEvaluationOptions) -> Segment:
    return Segment(
        tokens=tokens,
        metadata=SegmentMetadata(
            scale_root=config.scale_root,
            scale_type=config.scale_type,
            tokenization_context=tokenization_context_from_scale(
                scale_root=config.scale_root,
                scale_type=config.scale_type,
            ),
            time_numerator=config.time_numerator,
            time_denominator=config.time_denominator,
            bar_count=config.bar_count,
            window_start_bar=0,
            source_file=Path("generation-evaluation"),
            difficulty_level=None,
        ),
    )


def minimum_duration(config: GenerationEvaluationOptions) -> Fraction | None:
    if config.minimum_duration_denominator is None:
        return None

    return Fraction(1, config.minimum_duration_denominator)


def scale_type_to_id(scale_type: ScaleType) -> int:
    return tuple(ScaleType).index(scale_type)


def bar_positions(token_ids: list[int], *, token_vocabulary: TokenVocabulary) -> list[int]:
    positions = [0]
    bar_index = 0
    for token in token_vocabulary.decode(token_ids):
        positions.append(bar_index)
        if isinstance(token, BarToken):
            bar_index += 1

    return positions


def sample_token_id(
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


def constraint_report(
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
