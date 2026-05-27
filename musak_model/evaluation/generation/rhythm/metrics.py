import logging
from collections import Counter
from pathlib import Path
from time import perf_counter

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.evaluation.generation.protocols import GenerationEvaluationOptions
from musak_model.evaluation.generation.schema import GenerationSample
from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.profile.loading import FigureProfileArtifacts
from musak_model.n_grams.profile.rhythm.extraction import count_segment_rhythm_metrics
from musak_model.n_grams.profile.rhythm.loading import RhythmProfileArtifacts
from musak_model.n_grams.profile.rhythm.metrics import rhythm_reference_distribution_metrics
from musak_model.n_grams.profile.rhythm.schema import RhythmCountCounter
from musak_model.tokens.duration import DurationVocabulary

_LOGGER = logging.getLogger(__name__)


def rhythm_profile_metrics(
    artifacts: FigureProfileArtifacts | None,
    *,
    samples: list[GenerationSample],
    config: GenerationEvaluationOptions,
    analysis_config: NGramAnalysisConfig,
    duration_vocabulary: DurationVocabulary,
) -> dict[str, float]:
    if artifacts is None or artifacts.rhythm is None:
        return {}

    _LOGGER.info("Computing generation rhythm profile metrics: generated_samples=%s", len(samples))
    started_at = perf_counter()
    generated_counts, generated_sample_count = generated_rhythm_counts(
        samples,
        config=config,
        analysis_config=analysis_config,
        duration_vocabulary=duration_vocabulary,
    )
    metrics = {
        **rhythm_profile_count_metrics(artifacts.rhythm, generated_sample_count=generated_sample_count),
        **rhythm_reference_distribution_metrics(
            reference_counts=artifacts.rhythm.counts,
            comparison_counts=generated_counts,
            metric_prefix="generation/rhythm",
        ),
    }
    _LOGGER.info(
        "Computed generation rhythm profile metrics in %.1fs: counted_samples=%s",
        perf_counter() - started_at,
        generated_sample_count,
    )
    return metrics


def generated_rhythm_counts(
    samples: list[GenerationSample],
    *,
    config: GenerationEvaluationOptions,
    analysis_config: NGramAnalysisConfig,
    duration_vocabulary: DurationVocabulary,
) -> tuple[RhythmCountCounter, int]:
    counts: RhythmCountCounter = Counter()
    counted_sample_count = 0
    for sample in samples:
        if sample.decode_error is not None:
            continue

        try:
            counts.update(
                count_segment_rhythm_metrics(
                    _sample_segment(sample, config=config),
                    duration_vocabulary=duration_vocabulary,
                    rhythm_min_n=analysis_config.rhythm_min_n,
                    rhythm_max_n=analysis_config.rhythm_max_n,
                    grid_alignment_denominators=analysis_config.grid_alignment_denominators,
                    strong_beat_offsets=analysis_config.strong_beat_offsets,
                )
            )
        except ValueError:
            continue

        counted_sample_count += 1

    return counts, counted_sample_count


def rhythm_profile_count_metrics(
    artifacts: RhythmProfileArtifacts,
    *,
    generated_sample_count: int,
) -> dict[str, float]:
    return {
        "generation/rhythm/count/profile_samples": float(artifacts.profile.metadata.sample_count),
        "generation/rhythm/count/profile_groups": float(len(artifacts.profile.groups)),
        "generation/rhythm/count/generated_profile_samples": float(generated_sample_count),
    }


def _sample_segment(sample: GenerationSample, *, config: GenerationEvaluationOptions) -> Segment:
    return Segment(
        tokens=sample.tokens,
        metadata=SegmentMetadata(
            scale_root=config.scale_root,
            scale_type=config.scale_type,
            time_numerator=config.time_numerator,
            time_denominator=config.time_denominator,
            bar_count=sample.completed_bars,
            window_start_bar=0,
            source_file=Path("generated"),
            difficulty_level=None,
        ),
    )
