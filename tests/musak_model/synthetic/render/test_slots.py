from fractions import Fraction

import pytest
from numpy.random import default_rng

from musak_model.harmony.diatonic import natural_triad
from musak_model.synthetic.render.slots import render_slots
from musak_model.synthetic.structure.meter import (
    MetricalGrammarConfig,
    MetricalLeafType,
    MetricalNode,
    MetricalTree,
    MetricalTreeSampler,
)
from musak_model.tokens.schema import ScaleType

_TONIC = natural_triad(ScaleType.MAJOR, 1)
_DOMINANT = natural_triad(ScaleType.MAJOR, 5)


def _two_half_note_bar() -> MetricalTree:
    first = MetricalNode(Fraction(0), Fraction(1, 2), 1.0, (), MetricalLeafType.SOUND)
    second = MetricalNode(Fraction(1, 2), Fraction(1, 2), 0.5, (), MetricalLeafType.REST)
    bar = MetricalNode(Fraction(0), Fraction(1), 1.0, (first, second))
    return MetricalTree(4, 4, (bar,))


def test_whole_bar_frontier_is_a_single_region() -> None:
    tree = _two_half_note_bar()

    slots = render_slots(tree, [_TONIC], slot_duration=Fraction(1))

    assert len(slots) == 1
    assert slots[0].chord == _TONIC
    assert slots[0].offset == Fraction(0)
    assert slots[0].duration == Fraction(1)
    assert slots[0].leaf_type == MetricalLeafType.SOUND


def test_finer_frontier_assigns_distinct_chords() -> None:
    tree = _two_half_note_bar()

    slots = render_slots(tree, [_TONIC, _DOMINANT], slot_duration=Fraction(1, 2))

    assert len(slots) == 2
    assert slots[0].chord == _TONIC
    assert slots[1].chord == _DOMINANT
    assert slots[0].duration == Fraction(1, 2)
    assert slots[0].weight == 1.0


def test_rejects_chord_count_not_matching_the_frontier() -> None:
    tree = _two_half_note_bar()

    with pytest.raises(ValueError, match="frontier has 2 slots but 1 chords"):
        render_slots(tree, [_TONIC], slot_duration=Fraction(1, 2))


def test_slots_cover_the_frontier_for_a_sampled_tree() -> None:
    config = MetricalGrammarConfig.load().model_copy(update={"subdivision_probability": 1.0, "subdivision_decay": 1.0})
    tree = MetricalTreeSampler(config=config).sample(
        time_numerator=4, time_denominator=4, bar_count=2, rng=default_rng(0)
    )
    frontier = tree.harmonic_frontier(Fraction(1, 4))
    frontier_chords = [_TONIC if index % 2 == 0 else _DOMINANT for index in range(len(frontier))]

    slots = render_slots(tree, frontier_chords, slot_duration=Fraction(1, 4))

    assert len(slots) == len(frontier)
    assert {slot.chord for slot in slots} == {_TONIC, _DOMINANT}
    assert sum((slot.duration for slot in slots), Fraction(0)) == tree.duration
