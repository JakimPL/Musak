from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from musak_model.harmony.schema import Chord
from musak_model.synthetic.structure.meter import MetricalLeafType, MetricalTree


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

    slots: list[RenderSlot] = []
    for node, chord in zip(frontier, frontier_chords, strict=True):
        for leaf in node.leaves():
            if leaf.leaf_type is None:
                raise ValueError("a metrical leaf is missing its leaf type")
            slots.append(
                RenderSlot(
                    offset=leaf.offset,
                    duration=leaf.duration,
                    weight=leaf.weight,
                    leaf_type=leaf.leaf_type,
                    chord=chord,
                )
            )

    return tuple(slots)
