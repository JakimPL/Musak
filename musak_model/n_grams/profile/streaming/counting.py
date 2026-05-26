from collections import Counter

from musak_model.n_grams.figure.builder import scale_size_for_type
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.n_grams.figure.signature import figure_signature_to_json, iter_figure_signatures_from_run
from musak_model.n_grams.profile.schema import FigureProfileGroup, FigureSampleCounts
from musak_model.n_grams.profile.streaming.schema import FigureCountCounter
from musak_model.n_grams.profile.streaming.totals import figure_group_totals
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise


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
    counts: FigureCountCounter = Counter()
    for hand, runs in runs_by_hand.items():
        for run in runs:
            for n, signature in iter_figure_signatures_from_run(
                run,
                min_n=min_n,
                max_n=max_n,
                scale_size=scale_size,
            ):
                counts[(sample.scale_type.value, hand.value, n, figure_signature_to_json(signature))] += 1

    return counts


def sample_profile_payload(
    sample_index: int,
    scale_type: ScaleType,
    counts: FigureCountCounter,
) -> str:
    groups: list[FigureProfileGroup] = []
    for (group_scale_type, hand, n), totals in sorted(figure_group_totals(counts).items()):
        groups.append(
            FigureProfileGroup(
                scale_type=ScaleType(group_scale_type),
                hand=Hand(hand),
                n=n,
                total=totals.total,
                monophonic=totals.monophonic,
                chords_only=totals.chords_only,
                in_scale=totals.in_scale,
            )
        )

    return FigureSampleCounts(sample_index=sample_index, scale_type=scale_type, groups=tuple(groups)).model_dump_json()
