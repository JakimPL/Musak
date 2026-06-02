from fractions import Fraction

from numpy.random import default_rng

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.render.motif import MotifFigure, MotifSchema
from musak_model.synthetic.render.variation import VariationOperator, apply_variation, sample_operator, vary_motif
from musak_model.synthetic.structure.form import VariationKind


def _ngram(step: int) -> FigureNGram:
    return FigureNGram(onsets=((((step, 0),), Fraction(1)),))


def _schema(*anchor_offsets: int) -> MotifSchema:
    return MotifSchema(
        tuple(
            MotifFigure(slot_index=index, figure=_ngram(index), anchor_offset=offset)
            for index, offset in enumerate(anchor_offsets)
        )
    )


def test_transpose_shifts_all_anchor_offsets_and_keeps_figures() -> None:
    schema = _schema(0, 2, 1)

    out = apply_variation(schema, VariationOperator.DIATONIC_TRANSPOSE, transpose=3)

    assert [figure.anchor_offset for figure in out.figures] == [3, 5, 4]
    assert [figure.figure for figure in out.figures] == [figure.figure for figure in schema.figures]


def test_invert_negates_contour() -> None:
    out = apply_variation(_schema(0, 2, -1), VariationOperator.INVERT)

    assert [figure.anchor_offset for figure in out.figures] == [0, -2, 1]


def test_retrograde_reverses_figures_and_rebases_contour() -> None:
    out = apply_variation(_schema(0, 2, 3), VariationOperator.RETROGRADE)

    assert [figure.anchor_offset for figure in out.figures] == [0, -1, -3]
    assert [figure.slot_index for figure in out.figures] == [0, 1, 2]
    assert [figure.figure for figure in out.figures] == [_ngram(2), _ngram(1), _ngram(0)]


def test_identity_returns_the_same_schema() -> None:
    schema = _schema(0, 1)

    assert apply_variation(schema, VariationOperator.IDENTITY) is schema


def test_same_with_zero_budget_is_identity() -> None:
    operator, transpose = sample_operator(
        VariationKind.SAME, variation_budget=0.0, maximum_transpose=4, rng=default_rng(0)
    )

    assert operator is VariationOperator.IDENTITY
    assert transpose == 0


def test_variant_picks_a_variant_operator() -> None:
    operator, _ = sample_operator(VariationKind.VARIANT, variation_budget=1.0, maximum_transpose=4, rng=default_rng(1))

    assert operator in {
        VariationOperator.DIATONIC_TRANSPOSE,
        VariationOperator.INVERT,
        VariationOperator.RETROGRADE,
    }


def test_vary_motif_is_deterministic_for_a_seed() -> None:
    schema = _schema(0, 2, 1)

    first = vary_motif(schema, VariationKind.VARIANT, variation_budget=1.0, maximum_transpose=3, rng=default_rng(5))
    second = vary_motif(schema, VariationKind.VARIANT, variation_budget=1.0, maximum_transpose=3, rng=default_rng(5))

    assert first == second
