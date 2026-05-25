from musak_model.analysis.n_grams.figure.counter import FigureNGramCountsByHand
from musak_model.analysis.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.analysis.n_grams.profile.schema import (
    FigureProfile,
    FigureProfileGroup,
    FigureProfileMetadata,
    FigureSampleCounts,
)
from musak_model.tokens.schema import ScaleType


def build_figure_profile(
    counts: FigureNGramCountsByScale,
    metadata: FigureProfileMetadata,
) -> FigureProfile:
    groups: list[FigureProfileGroup] = []
    for scale_type, counts_by_hand in sorted(counts.items(), key=lambda item: item[0].value):
        groups.extend(figure_profile_groups(scale_type=scale_type, counts_by_hand=counts_by_hand))

    return FigureProfile(metadata=metadata, groups=tuple(groups))


def build_figure_sample_counts(
    *,
    sample_index: int,
    scale_type: ScaleType,
    counts_by_hand: FigureNGramCountsByHand,
) -> FigureSampleCounts:
    return FigureSampleCounts(
        sample_index=sample_index,
        scale_type=scale_type,
        groups=tuple(figure_profile_groups(scale_type=scale_type, counts_by_hand=counts_by_hand)),
    )


def figure_profile_groups(
    *,
    scale_type: ScaleType,
    counts_by_hand: FigureNGramCountsByHand,
) -> tuple[FigureProfileGroup, ...]:
    groups: list[FigureProfileGroup] = []
    for hand, counts_by_n in sorted(counts_by_hand.items(), key=lambda item: item[0].value):
        for n, figure_counts in sorted(counts_by_n.items()):
            groups.append(
                FigureProfileGroup(
                    scale_type=scale_type,
                    hand=hand,
                    n=n,
                    total=sum(figure_counts.values()),
                    monophonic=sum(count for figure, count in figure_counts.items() if figure.monophonic),
                    chords_only=sum(count for figure, count in figure_counts.items() if figure.chords_only),
                    in_scale=sum(count for figure, count in figure_counts.items() if figure.in_scale),
                )
            )

    return tuple(groups)
