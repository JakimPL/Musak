from fractions import Fraction
from typing import Final

from numpy.random import default_rng

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.figures import FigureVocabularyEntry, FigureVocabularyGroup
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.motif import MotifSlot, select_motif_seed
from musak_model.tokens.schema import Hand, ScaleType

_CHORD_PITCH_CLASSES: Final = frozenset({0, 4, 7})


def _ngram(*steps: int) -> FigureNGram:
    return FigureNGram(onsets=tuple((((step, 0),), Fraction(1)) for step in steps))


def _entry(figure: FigureNGram, count: int) -> FigureVocabularyEntry:
    group = FigureVocabularyGroup(scale_type=ScaleType.MAJOR, hand=Hand.RIGHT, n=len(figure.onsets))
    return FigureVocabularyEntry(group=group, figure=figure, count=count)


_ENTRIES = (_entry(_ngram(0), 5), _entry(_ngram(0, 1), 3), _entry(_ngram(0, 2), 2))


def _slot(slot_index: int, anchor: int, entries: tuple[FigureVocabularyEntry, ...] = _ENTRIES) -> MotifSlot:
    return MotifSlot(
        slot_index=slot_index,
        anchor=anchor,
        target_slope=0,
        chord_pitch_classes=_CHORD_PITCH_CLASSES,
        weight=1.0,
        entries=entries,
    )


def test_select_motif_seed_returns_one_figure_per_sound_slot() -> None:
    slots = [_slot(0, 0), _slot(2, 2), _slot(3, 1)]

    schema = select_motif_seed(
        slots, scale_type=ScaleType.MAJOR, config=RenderConfig.load(), candidate_count=4, rng=default_rng(0)
    )

    assert schema is not None
    assert [figure.slot_index for figure in schema.figures] == [0, 2, 3]
    assert schema.figures[0].anchor_offset == 0
    assert schema.figures[1].anchor_offset == 2
    assert schema.sound_slot_count == 3


def test_select_motif_seed_is_deterministic_for_a_seed() -> None:
    slots = [_slot(0, 0), _slot(1, 1)]

    first = select_motif_seed(
        slots, scale_type=ScaleType.MAJOR, config=RenderConfig.load(), candidate_count=3, rng=default_rng(7)
    )
    second = select_motif_seed(
        slots, scale_type=ScaleType.MAJOR, config=RenderConfig.load(), candidate_count=3, rng=default_rng(7)
    )

    assert first == second


def test_select_motif_seed_returns_none_without_feasible_slots() -> None:
    slots = [_slot(0, 0, entries=())]

    schema = select_motif_seed(
        slots, scale_type=ScaleType.MAJOR, config=RenderConfig.load(), candidate_count=2, rng=default_rng(0)
    )

    assert schema is None
