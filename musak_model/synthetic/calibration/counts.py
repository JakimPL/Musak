from collections.abc import Sequence

from musak_model.data.schema import Segment
from musak_model.n_grams.figure.counter import count_hand_figure_ngrams
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.n_grams.figure.samples.merge import merge_scale_counts
from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import scale_size_for_type


def segment_figure_counts(
    segments: Sequence[Segment],
    *,
    min_n: int,
    max_n: int,
    duration_vocabulary: DurationVocabulary,
) -> FigureNGramCountsByScale:
    counts_by_scale: FigureNGramCountsByScale = {}
    for segment in segments:
        runs_by_hand = extract_hand_onset_runs(
            segment.tokens,
            duration_vocabulary=duration_vocabulary,
            time_numerator=segment.metadata.time_numerator,
            time_denominator=segment.metadata.time_denominator,
        )
        sample_counts = count_hand_figure_ngrams(
            runs_by_hand,
            min_n=min_n,
            max_n=max_n,
            scale_size=scale_size_for_type(segment.metadata.scale_type),
        )
        merge_scale_counts(
            counts_by_scale,
            scale_type=segment.metadata.scale_type,
            sample_counts=sample_counts,
        )

    return counts_by_scale
