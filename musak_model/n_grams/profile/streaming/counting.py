from collections import Counter
from fractions import Fraction

from musak_model.n_grams.figure.builder import scale_size_for_type
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.n_grams.figure.signature import figure_signature_to_json, iter_figure_occurrences_from_run
from musak_model.n_grams.profile.schema import FigureProfileGroup, FigureSampleCounts
from musak_model.n_grams.profile.streaming.schema import FigureCountCounter, FigureCountKey
from musak_model.n_grams.profile.streaming.totals import figure_group_totals
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise
from musak_shared.ratios import format_ratio


def count_sample_figure_signatures(
    sample: EncodedExercise,
    *,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
    min_n: int,
    max_n: int,
) -> FigureCountCounter:
    tokens = token_vocabulary.decode(sample.token_ids)
    runs_by_hand = extract_hand_onset_runs(
        tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=sample.time_numerator,
        time_denominator=sample.time_denominator,
    )
    scale_size = scale_size_for_type(sample.scale_type)
    measure_duration = Fraction(sample.time_numerator, sample.time_denominator)
    time_signature = format_ratio((sample.time_numerator, sample.time_denominator))
    counts: FigureCountCounter = Counter()
    for hand, runs in runs_by_hand.items():
        for run in runs:
            for occurrence in iter_figure_occurrences_from_run(
                run,
                min_n=min_n,
                max_n=max_n,
                scale_size=scale_size,
            ):
                key = FigureCountKey(
                    scale_type=sample.scale_type.value,
                    hand=hand.value,
                    figure_length=occurrence.figure_length,
                    figure=figure_signature_to_json(occurrence.signature),
                    anchor_degree=occurrence.anchor_degree,
                    anchor_accidental=occurrence.anchor_accidental,
                    anchor_octave=occurrence.anchor_octave,
                    base_duration=format_ratio(occurrence.base_duration),
                    bar_relative_onset=format_ratio(occurrence.start % measure_duration),
                    time_signature=time_signature,
                )
                counts[key] += 1

    return counts


def sample_profile_payload(
    sample_index: int,
    scale_type: ScaleType,
    counts: FigureCountCounter,
) -> str:
    groups: list[FigureProfileGroup] = []
    for group_key, totals in sorted(figure_group_totals(counts).items()):
        groups.append(
            FigureProfileGroup(
                scale_type=ScaleType(group_key.scale_type),
                hand=Hand(group_key.hand),
                n=group_key.figure_length,
                total=totals.total,
                monophonic=totals.monophonic,
                chords_only=totals.chords_only,
                in_scale=totals.in_scale,
            )
        )

    return FigureSampleCounts(sample_index=sample_index, scale_type=scale_type, groups=tuple(groups)).model_dump_json()
