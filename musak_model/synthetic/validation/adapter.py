from __future__ import annotations

from musak_model.data.schema import Segment
from musak_model.evaluation.diagnostics import SegmentDiagnostics, diagnose_segment
from musak_model.evaluation.generation.sampling import constraint_report, constraints_from_config
from musak_model.evaluation.generation.schema import GenerationSample
from musak_model.synthetic.validation.options import MetricOptions
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import BarToken, EndToken


def generation_sample(
    segment: Segment,
    *,
    options: MetricOptions,
    duration_vocabulary: DurationVocabulary,
) -> GenerationSample:
    diagnostics: SegmentDiagnostics | None = None
    decode_error: str | None = None
    try:
        diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)
    except ValueError as exception:
        decode_error = str(exception)

    return GenerationSample(
        tokens=segment.tokens,
        reached_end=bool(segment.tokens and isinstance(segment.tokens[-1], EndToken)),
        generated_token_count=len(segment.tokens),
        constraint_error=None,
        constraint_report=constraint_report(
            segment.tokens, constraints=constraints_from_config(options), duration_vocabulary=duration_vocabulary
        ),
        diagnostics=diagnostics,
        decode_error=decode_error,
        completed_bars=sum(isinstance(token, BarToken) for token in segment.tokens),
        target_bar_count=options.bar_count,
    )
