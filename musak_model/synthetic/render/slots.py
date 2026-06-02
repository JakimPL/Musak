from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from musak_model.harmony.schema import Chord
from musak_model.synthetic.structure.meter import MetricalLeafType, MetricalNode, MetricalTree


@dataclass(frozen=True)
class RenderSlot:
    offset: Fraction
    duration: Fraction
    weight: float
    leaf_type: MetricalLeafType
    chord: Chord


def render_slots(
    metrical_tree: MetricalTree,
    frontier_chords: Sequence[Chord],
    *,
    slot_duration: Fraction,
) -> tuple[RenderSlot, ...]:
    frontier = metrical_tree.harmonic_frontier(slot_duration)
    if len(frontier) != len(frontier_chords):
        raise ValueError(f"frontier has {len(frontier)} slots but {len(frontier_chords)} chords were given")

    return tuple(
        RenderSlot(
            offset=node.offset,
            duration=node.duration,
            weight=node.weight,
            leaf_type=_region_activity(node),
            chord=chord,
        )
        for node, chord in zip(frontier, frontier_chords, strict=True)
    )


def _region_activity(node: MetricalNode) -> MetricalLeafType:
    if all(leaf.leaf_type is MetricalLeafType.REST for leaf in node.leaves()):
        return MetricalLeafType.REST

    return MetricalLeafType.SOUND
