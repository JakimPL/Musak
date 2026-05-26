from collections import Counter
from collections.abc import Iterable

from musak_model.n_grams.figure.signature import (
    figure_signature_chords_only,
    figure_signature_from_json,
    figure_signature_in_scale,
    figure_signature_monophonic,
)
from musak_model.n_grams.profile.streaming.schema import (
    FigureCountCounter,
    FigureCountKey,
    FigureGroupTotals,
    FigureGroupTotalsByKey,
)


def figure_group_totals(counts: Iterable[tuple[FigureCountKey, int]] | FigureCountCounter) -> FigureGroupTotalsByKey:
    items = counts.items() if isinstance(counts, Counter) else counts
    totals_by_group: FigureGroupTotalsByKey = {}
    for (scale_type, hand, n, figure_json), count in items:
        signature = figure_signature_from_json(figure_json)
        key = (scale_type, hand, n)
        totals = totals_by_group.get(key, FigureGroupTotals(total=0, monophonic=0, chords_only=0, in_scale=0))
        monophonic = totals.monophonic
        chords_only = totals.chords_only
        in_scale = totals.in_scale
        if figure_signature_monophonic(signature):
            monophonic += count
        if figure_signature_chords_only(signature):
            chords_only += count
        if figure_signature_in_scale(signature):
            in_scale += count

        totals_by_group[key] = FigureGroupTotals(
            total=totals.total + count,
            monophonic=monophonic,
            chords_only=chords_only,
            in_scale=in_scale,
        )

    return totals_by_group
