from musak_model.analysis.n_grams.figure.encoded import FigureNGramCountsByScale
from musak_model.analysis.n_grams.profile.schema import (
    FigureProfile,
    FigureProfileGroup,
    FigureProfileMetadata,
)


def build_figure_profile(
    counts: FigureNGramCountsByScale,
    metadata: FigureProfileMetadata,
) -> FigureProfile:
    groups: list[FigureProfileGroup] = []
    for scale_type, counts_by_hand in sorted(counts.items(), key=lambda item: item[0].value):
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

    return FigureProfile(metadata=metadata, groups=tuple(groups))
