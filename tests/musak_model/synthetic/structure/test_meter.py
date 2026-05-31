from dataclasses import dataclass
from fractions import Fraction

import pytest
from numpy.random import default_rng

from musak_model.synthetic.processes.accent import indispensability_per_position
from musak_model.synthetic.structure.meter import (
    MetricalGrammarConfig,
    MetricalLeafType,
    MetricalNode,
    MetricalTree,
    MetricalTreeSampler,
    metrical_factors,
)


@dataclass(frozen=True)
class _FactorCase:
    time_numerator: int
    time_denominator: int
    expected: tuple[int, ...]


_FACTOR_CASES = (
    _FactorCase(4, 4, (2, 2)),
    _FactorCase(2, 4, (2,)),
    _FactorCase(3, 4, (3,)),
    _FactorCase(6, 8, (2, 3)),
    _FactorCase(9, 8, (3, 3)),
    _FactorCase(12, 8, (2, 2, 3)),
    _FactorCase(5, 4, (5,)),
    _FactorCase(3, 8, (3,)),
)


@pytest.mark.parametrize("case", _FACTOR_CASES, ids=lambda case: f"{case.time_numerator}/{case.time_denominator}")
def test_metrical_factors_derive_from_meter(case: _FactorCase) -> None:
    assert metrical_factors(case.time_numerator, case.time_denominator) == case.expected


def _default_sampler() -> MetricalTreeSampler:
    return MetricalTreeSampler(config=MetricalGrammarConfig.load())


def _fully_subdivided_sampler() -> MetricalTreeSampler:
    config = MetricalGrammarConfig.load().model_copy(update={"subdivision_probability": 1.0, "subdivision_decay": 1.0})
    return MetricalTreeSampler(config=config)


def test_full_dyadic_tree_weights_match_gcd_indispensability() -> None:
    tree = _fully_subdivided_sampler().sample(time_numerator=4, time_denominator=4, bar_count=1, rng=default_rng(0))
    indispensability = indispensability_per_position(16)

    leaves = tree.leaves()

    assert len(leaves) == 16
    for position, leaf in enumerate(leaves):
        assert leaf.offset == Fraction(position, 16)
        assert leaf.weight == pytest.approx(float(indispensability[position]))


def test_compound_meter_follows_subdivision_not_gcd() -> None:
    tree = _fully_subdivided_sampler().sample(time_numerator=6, time_denominator=8, bar_count=1, rng=default_rng(0))
    weight_at = {leaf.offset: leaf.weight for leaf in tree.leaves()}

    assert weight_at[Fraction(3, 8)] == pytest.approx(0.5)
    assert weight_at[Fraction(1, 8)] == pytest.approx(1 / 6)
    assert weight_at[Fraction(3, 8)] > weight_at[Fraction(1, 8)]


def test_leaves_tile_the_segment_contiguously() -> None:
    tree = _default_sampler().sample(time_numerator=4, time_denominator=4, bar_count=3, rng=default_rng(1))
    leaves = tree.leaves()

    assert leaves[0].offset == Fraction(0)
    expected_offset = Fraction(0)
    for leaf in leaves:
        assert leaf.offset == expected_offset
        expected_offset += leaf.duration
    assert expected_offset == tree.duration == Fraction(3)


def test_sampler_is_deterministic_for_a_seed() -> None:
    sampler = _default_sampler()

    first = sampler.sample(time_numerator=4, time_denominator=4, bar_count=2, rng=default_rng(7))
    second = sampler.sample(time_numerator=4, time_denominator=4, bar_count=2, rng=default_rng(7))

    assert first == second


def test_harmonic_frontier_covers_the_span_at_quarter_granularity() -> None:
    tree = _fully_subdivided_sampler().sample(time_numerator=4, time_denominator=4, bar_count=2, rng=default_rng(0))

    frontier = tree.harmonic_frontier(Fraction(1, 4))

    assert len(frontier) == 8
    assert all(node.duration == Fraction(1, 4) for node in frontier)
    assert sum((node.duration for node in frontier), Fraction(0)) == tree.duration


def test_harmonic_frontier_always_covers_the_span_for_a_stochastic_tree() -> None:
    tree = _default_sampler().sample(time_numerator=3, time_denominator=4, bar_count=2, rng=default_rng(3))

    frontier = tree.harmonic_frontier(Fraction(1, 4))

    assert sum((node.duration for node in frontier), Fraction(0)) == tree.duration
    assert all(node.duration <= Fraction(1, 4) or node.is_leaf for node in frontier)


def test_whole_note_and_cross_bar_tie_are_expressible() -> None:
    whole = Fraction(1)
    struck_bar = MetricalNode(Fraction(0), whole, 1.0, (), MetricalLeafType.SOUND)
    held_bar = MetricalNode(whole, whole, 1.0, (), MetricalLeafType.TIE)

    tree = MetricalTree(4, 4, (struck_bar, held_bar))

    assert tree.leaves() == (struck_bar, held_bar)
    assert struck_bar.leaf_type is MetricalLeafType.SOUND
    assert held_bar.leaf_type is MetricalLeafType.TIE
    assert tree.duration == Fraction(2)


def test_sample_produces_only_leaf_types_in_the_enum() -> None:
    tree = _default_sampler().sample(time_numerator=4, time_denominator=4, bar_count=4, rng=default_rng(5))

    assert all(leaf.leaf_type in set(MetricalLeafType) for leaf in tree.leaves())


def test_node_rejects_children_that_do_not_tile_the_span() -> None:
    short_child = MetricalNode(Fraction(0), Fraction(1, 4), 1.0, (), MetricalLeafType.SOUND)

    with pytest.raises(ValueError, match="sum to the node duration"):
        MetricalNode(Fraction(0), Fraction(1, 2), 1.0, (short_child,), None)


def test_node_rejects_weight_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="weight"):
        MetricalNode(Fraction(0), Fraction(1), 0.0, (), MetricalLeafType.SOUND)


def test_leaf_requires_a_leaf_type() -> None:
    with pytest.raises(ValueError, match="leaf"):
        MetricalNode(Fraction(0), Fraction(1), 1.0, (), None)


def test_sample_rejects_non_positive_dimensions() -> None:
    sampler = _default_sampler()

    with pytest.raises(ValueError, match="bar_count"):
        sampler.sample(time_numerator=4, time_denominator=4, bar_count=0, rng=default_rng(0))

    with pytest.raises(ValueError, match="time signature"):
        sampler.sample(time_numerator=0, time_denominator=4, bar_count=1, rng=default_rng(0))
