from collections import Counter
from fractions import Fraction

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.fitting.form.analysis import AnalyzedPiece
from musak_model.synthetic.fitting.form.repetition import RepetitionConfig, analyze_repetition
from musak_model.tokens.schema import ScaleType

_CONFIG = RepetitionConfig(segment_length_candidates=(1, 2, 4), similarity_bucket_count=20)


def _ngram(step: int) -> FigureNGram:
    return FigureNGram(onsets=((((step, 0),), Fraction(1)),))


def _bar(*steps: int) -> Counter[FigureNGram]:
    return Counter(_ngram(step) for step in steps)


def _piece(bar_figures: list[Counter[FigureNGram]]) -> AnalyzedPiece:
    return AnalyzedPiece(
        scale_type=ScaleType.MAJOR,
        bar_count=len(bar_figures),
        bar_duration=Fraction(1),
        slots=(),
        bar_figures=tuple(bar_figures),
    )


def test_periodic_piece_prefers_the_matching_segment_length() -> None:
    analysis = analyze_repetition(_piece([_bar(0, 1), _bar(2, 3), _bar(0, 1), _bar(2, 3)]), config=_CONFIG)

    assert analysis is not None
    assert analysis.segment_length == 2
    assert max(analysis.best_match_similarities) == 1.0


def test_distinct_segments_have_zero_similarity() -> None:
    config = RepetitionConfig(segment_length_candidates=(1,), similarity_bucket_count=20)

    analysis = analyze_repetition(_piece([_bar(0), _bar(5)]), config=config)

    assert analysis is not None
    assert analysis.best_match_similarities == (0.0,)


def test_returns_none_without_enough_segments() -> None:
    config = RepetitionConfig(segment_length_candidates=(4,), similarity_bucket_count=20)

    assert analyze_repetition(_piece([_bar(0), _bar(1)]), config=config) is None
