from typing import Final

from musak_model.evaluation.coherence import coherence_metrics
from musak_model.evaluation.generation.protocols import GenerationEvaluationOptions
from musak_model.evaluation.generation.sampling import segment_from_tokens
from musak_model.evaluation.generation.schema import GenerationSample
from musak_model.tokens.duration import DurationVocabulary

_METRIC_GROUP_NAME: Final[str] = "coherence"


def coherence_profile_metrics(
    suite_name: str,
    samples: list[GenerationSample],
    *,
    config: GenerationEvaluationOptions,
    duration_vocabulary: DurationVocabulary,
) -> dict[str, float]:
    segments = [
        segment_from_tokens(
            sample.tokens,
            config=config,
        )
        for sample in samples
        if sample.decode_error is None
    ]
    return coherence_metrics(
        segments,
        duration_vocabulary=duration_vocabulary,
        metric_prefix=f"generation/{suite_name}/{_METRIC_GROUP_NAME}",
    )
