import logging
from dataclasses import dataclass
from time import perf_counter

from musak_model.evaluation.generation.protocols import GenerationEvaluationOptions
from musak_model.evaluation.generation.schema import GenerationSample
from musak_model.n_grams.figure.counter import FigureNGramCountsByHand, count_hand_figure_ngrams
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.n_grams.figure.samples.merge import merge_scale_counts
from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.profile.builder import build_figure_profile
from musak_model.n_grams.profile.loading import FigureProfileArtifacts
from musak_model.n_grams.profile.metrics.distribution import figure_distribution_metrics
from musak_model.n_grams.profile.metrics.profile_comparison import figure_profile_comparison_metrics
from musak_model.n_grams.profile.schema import FigureProfile, FigureProfileMetadata
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import scale_size_for_type

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GeneratedFigureArtifacts:
    counts_by_scale: FigureNGramCountsByScale
    profile: FigureProfile
    sample_count: int


def figure_profile_metrics(
    artifacts: FigureProfileArtifacts | None,
    *,
    samples: list[GenerationSample],
    config: GenerationEvaluationOptions,
    duration_vocabulary: DurationVocabulary,
) -> dict[str, float]:
    if artifacts is None:
        return {}

    _LOGGER.info("Computing generation figure profile metrics: generated_samples=%s", len(samples))
    started_at = perf_counter()
    generated_artifacts = _generated_figure_artifacts(
        samples,
        reference_profile=artifacts.profile,
        config=config,
        duration_vocabulary=duration_vocabulary,
    )
    metrics = _figure_comparison_metrics(artifacts, generated_artifacts=generated_artifacts)
    _LOGGER.info(
        "Computed generation figure profile metrics in %.1fs: profile_samples=%s",
        perf_counter() - started_at,
        generated_artifacts.sample_count,
    )
    return metrics


def _generated_figure_artifacts(
    samples: list[GenerationSample],
    *,
    reference_profile: FigureProfile,
    config: GenerationEvaluationOptions,
    duration_vocabulary: DurationVocabulary,
) -> _GeneratedFigureArtifacts:
    generated_counts_by_scale, generated_sample_count = generated_figure_counts(
        samples,
        reference_profile=reference_profile,
        config=config,
        duration_vocabulary=duration_vocabulary,
    )
    return _GeneratedFigureArtifacts(
        counts_by_scale=generated_counts_by_scale,
        profile=generated_figure_profile(
            generated_counts_by_scale,
            reference_profile=reference_profile,
            sample_count=generated_sample_count,
        ),
        sample_count=generated_sample_count,
    )


def _figure_comparison_metrics(
    artifacts: FigureProfileArtifacts,
    *,
    generated_artifacts: _GeneratedFigureArtifacts,
) -> dict[str, float]:
    return {
        **figure_profile_count_metrics(artifacts),
        **figure_profile_comparison_metrics(
            reference_profile=artifacts.profile,
            comparison_profile=generated_artifacts.profile,
            metric_prefix="generation/figure",
            comparison_sample_count_metric="generated_profile_samples",
        ),
        **figure_distribution_metrics(
            reference_counts=artifacts.counts_by_scale,
            comparison_counts=generated_artifacts.counts_by_scale,
            metric_prefix="generation/figure",
        ),
    }


def figure_profile_count_metrics(artifacts: FigureProfileArtifacts) -> dict[str, float]:
    return {
        "generation/figure/count/profile_samples": float(artifacts.profile.metadata.sample_count),
        "generation/figure/count/profile_groups": float(len(artifacts.profile.groups)),
        "generation/figure/count/sample_profiles": float(len(artifacts.sample_counts)),
    }


def generated_figure_profile(
    counts_by_scale: FigureNGramCountsByScale,
    *,
    reference_profile: FigureProfile,
    sample_count: int,
) -> FigureProfile:
    return build_figure_profile(
        counts_by_scale,
        FigureProfileMetadata(
            min_n=reference_profile.metadata.min_n,
            max_n=reference_profile.metadata.max_n,
            sample_count=sample_count,
        ),
    )


def generated_figure_counts(
    samples: list[GenerationSample],
    *,
    reference_profile: FigureProfile,
    config: GenerationEvaluationOptions,
    duration_vocabulary: DurationVocabulary,
) -> tuple[FigureNGramCountsByScale, int]:
    counts_by_scale: FigureNGramCountsByScale = {}
    counted_sample_count = 0
    for sample in samples:
        counts_by_hand = _sample_figure_counts(
            sample,
            reference_profile=reference_profile,
            config=config,
            duration_vocabulary=duration_vocabulary,
        )
        if counts_by_hand is None:
            continue

        merge_scale_counts(counts_by_scale, scale_type=config.scale_type, sample_counts=counts_by_hand)
        counted_sample_count += 1

    return counts_by_scale, counted_sample_count


def _sample_figure_counts(
    sample: GenerationSample,
    *,
    reference_profile: FigureProfile,
    config: GenerationEvaluationOptions,
    duration_vocabulary: DurationVocabulary,
) -> FigureNGramCountsByHand | None:
    if sample.decode_error is not None:
        return None

    try:
        runs_by_hand = extract_hand_onset_runs(
            sample.tokens,
            duration_vocabulary=duration_vocabulary,
            time_numerator=config.time_numerator,
            time_denominator=config.time_denominator,
        )
        return count_hand_figure_ngrams(
            runs_by_hand,
            min_n=reference_profile.metadata.min_n,
            max_n=reference_profile.metadata.max_n,
            scale_size=scale_size_for_type(config.scale_type),
        )
    except ValueError:
        return None
