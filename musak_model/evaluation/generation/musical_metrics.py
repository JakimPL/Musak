from typing import Final

from musak_model.evaluation.generation.protocols import GenerationEvaluationOptions
from musak_model.evaluation.generation.sampling import segment_from_tokens
from musak_model.evaluation.generation.schema import GenerationSample
from musak_model.evaluation.musical import musical_metrics
from musak_model.tokens.duration import DurationVocabulary

_METRIC_PREFIX: Final[str] = "generation/musical"


def musical_profile_metrics(
    *,
    samples: list[GenerationSample],
    config: GenerationEvaluationOptions,
    duration_vocabulary: DurationVocabulary,
) -> dict[str, float]:
    segments = [segment_from_tokens(sample.tokens, config=config) for sample in samples if sample.decode_error is None]
    return musical_metrics(
        segments,
        duration_vocabulary=duration_vocabulary,
        metric_prefix=_METRIC_PREFIX,
    )
