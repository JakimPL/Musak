from enum import StrEnum
from typing import Final

from numpy.random import Generator

from musak_model.synthetic.render.motif import MotifFigure, MotifSchema
from musak_model.synthetic.structure.form import VariationKind


class VariationOperator(StrEnum):
    IDENTITY = "identity"
    DIATONIC_TRANSPOSE = "diatonic_transpose"
    INVERT = "invert"
    RETROGRADE = "retrograde"


_VARIANT_OPERATORS: Final[tuple[VariationOperator, ...]] = (
    VariationOperator.DIATONIC_TRANSPOSE,
    VariationOperator.INVERT,
    VariationOperator.RETROGRADE,
)
_DESCENDING_SIGN_PROBABILITY: Final[float] = 0.5


def vary_motif(
    schema: MotifSchema,
    variation: VariationKind,
    *,
    variation_budget: float,
    maximum_transpose: int,
    rng: Generator,
) -> MotifSchema:
    operator, transpose = sample_operator(
        variation, variation_budget=variation_budget, maximum_transpose=maximum_transpose, rng=rng
    )
    return apply_variation(schema, operator, transpose=transpose)


def sample_operator(
    variation: VariationKind,
    *,
    variation_budget: float,
    maximum_transpose: int,
    rng: Generator,
) -> tuple[VariationOperator, int]:
    if variation is VariationKind.SAME:
        if rng.random() < variation_budget:
            return VariationOperator.DIATONIC_TRANSPOSE, _sample_transpose(maximum_transpose, rng)

        return VariationOperator.IDENTITY, 0

    if variation is VariationKind.VARIANT:
        operator = _VARIANT_OPERATORS[int(rng.integers(0, len(_VARIANT_OPERATORS)))]
        transpose = _sample_transpose(maximum_transpose, rng) if operator is VariationOperator.DIATONIC_TRANSPOSE else 0
        return operator, transpose

    return VariationOperator.IDENTITY, 0


def apply_variation(schema: MotifSchema, operator: VariationOperator, *, transpose: int = 0) -> MotifSchema:
    match operator:
        case VariationOperator.IDENTITY:
            return schema
        case VariationOperator.DIATONIC_TRANSPOSE:
            return _transpose(schema, transpose)
        case VariationOperator.INVERT:
            return _invert(schema)
        case VariationOperator.RETROGRADE:
            return _retrograde(schema)


def _transpose(schema: MotifSchema, delta: int) -> MotifSchema:
    return MotifSchema(
        tuple(
            MotifFigure(slot_index=figure.slot_index, figure=figure.figure, anchor_offset=figure.anchor_offset + delta)
            for figure in schema.figures
        )
    )


def _invert(schema: MotifSchema) -> MotifSchema:
    return MotifSchema(
        tuple(
            MotifFigure(slot_index=figure.slot_index, figure=figure.figure, anchor_offset=-figure.anchor_offset)
            for figure in schema.figures
        )
    )


def _retrograde(schema: MotifSchema) -> MotifSchema:
    slot_indices = [figure.slot_index for figure in schema.figures]
    reversed_figures = list(reversed(schema.figures))
    base_offset = reversed_figures[0].anchor_offset
    return MotifSchema(
        tuple(
            MotifFigure(
                slot_index=slot_index,
                figure=source.figure,
                anchor_offset=source.anchor_offset - base_offset,
            )
            for slot_index, source in zip(slot_indices, reversed_figures, strict=True)
        )
    )


def _sample_transpose(maximum_transpose: int, rng: Generator) -> int:
    if maximum_transpose <= 0:
        return 0

    magnitude = int(rng.integers(1, maximum_transpose + 1))
    return -magnitude if rng.random() < _DESCENDING_SIGN_PROBABILITY else magnitude
