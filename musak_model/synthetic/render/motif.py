from collections.abc import Sequence
from dataclasses import dataclass

from numpy.random import Generator

from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.synthetic.figures import FigureVocabularyEntry
from musak_model.synthetic.render.config import RenderConfig
from musak_model.synthetic.render.figure_selection import select_scored_figure
from musak_model.tokens.schema import ScaleType


@dataclass(frozen=True)
class MotifFigure:
    slot_index: int
    figure: FigureNGram
    anchor_offset: int


@dataclass(frozen=True)
class MotifSchema:
    figures: tuple[MotifFigure, ...]

    @property
    def sound_slot_count(self) -> int:
        return len(self.figures)


@dataclass(frozen=True)
class GroundedMotifFigure:
    anchor: int
    figure: FigureNGram


@dataclass(frozen=True)
class MotifSlot:
    slot_index: int
    anchor: int
    target_slope: int
    chord_pitch_classes: frozenset[int]
    weight: float
    entries: tuple[FigureVocabularyEntry, ...]


def select_motif_seed(
    slots: Sequence[MotifSlot],
    *,
    scale_type: ScaleType,
    config: RenderConfig,
    candidate_count: int,
    rng: Generator,
) -> MotifSchema | None:
    feasible = [slot for slot in slots if slot.entries]
    if not feasible:
        return None

    base_anchor = feasible[0].anchor
    best_schema: MotifSchema | None = None
    best_quality = float("-inf")
    for _ in range(max(1, candidate_count)):
        figures, quality = _render_candidate(
            feasible,
            base_anchor=base_anchor,
            scale_type=scale_type,
            config=config,
            rng=rng,
        )
        if quality > best_quality:
            best_quality = quality
            best_schema = MotifSchema(figures=tuple(figures))

    return best_schema


def ground_motif(
    schema: MotifSchema,
    *,
    base_anchor: int,
) -> dict[int, GroundedMotifFigure]:
    return {
        figure.slot_index: GroundedMotifFigure(
            anchor=base_anchor + figure.anchor_offset,
            figure=figure.figure,
        )
        for figure in schema.figures
    }


def _render_candidate(
    slots: Sequence[MotifSlot],
    *,
    base_anchor: int,
    scale_type: ScaleType,
    config: RenderConfig,
    rng: Generator,
) -> tuple[list[MotifFigure], float]:
    figures: list[MotifFigure] = []
    total_score = 0.0
    for slot in slots:
        entry, score = select_scored_figure(
            slot.entries,
            anchor=slot.anchor,
            target_slope=slot.target_slope,
            scale_type=scale_type,
            chord_pitch_classes=slot.chord_pitch_classes,
            weight=slot.weight,
            config=config,
            rng=rng,
        )
        figures.append(
            MotifFigure(
                slot_index=slot.slot_index,
                figure=entry.figure,
                anchor_offset=slot.anchor - base_anchor,
            )
        )
        total_score += score

    return figures, total_score / len(slots)
