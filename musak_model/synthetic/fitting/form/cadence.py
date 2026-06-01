from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from pydantic import BaseModel, ConfigDict, Field

from musak_model.synthetic.fitting.form.analysis import AnalyzedPiece, HarmonicSlot
from musak_model.synthetic.structure.harmony_grammar import ClosingPattern
from musak_shared.elements import HarmonicFunction


class CadenceDetectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metrical_weight: float = Field(ge=0)
    harmonic_arrival_weight: float = Field(ge=0)
    rhythmic_stop_weight: float = Field(ge=0)
    bar_alignment_weight: float = Field(ge=0)
    cadence_threshold: float = Field(ge=0)
    minimum_cadence_separation_slots: int = Field(ge=0)
    maximum_closing_slots: int = Field(gt=0)


@dataclass(frozen=True)
class Cadence:
    arrival_slot_index: int
    end_bar: int
    closing: ClosingPattern
    is_final: bool
    boundary_score: float


def detect_cadences(
    piece: AnalyzedPiece,
    *,
    config: CadenceDetectionConfig,
) -> tuple[Cadence, ...]:
    slots = piece.slots
    if not slots:
        return ()

    scores = [
        _boundary_score(
            slots,
            index,
            config=config,
            bar_duration=piece.bar_duration,
        )
        for index in range(len(slots))
    ]
    final_index = len(slots) - 1
    selected = _select_arrivals(
        scores,
        threshold=config.cadence_threshold,
        separation=config.minimum_cadence_separation_slots,
        mandatory_index=final_index,
    )

    cadences: list[Cadence] = []
    for index in selected:
        closing = _closing_pattern(
            slots,
            index,
            maximum_closing_slots=config.maximum_closing_slots,
        )
        if closing is None:
            continue

        cadences.append(
            Cadence(
                arrival_slot_index=index,
                end_bar=_slot_end_bar(
                    slots[index],
                    bar_duration=piece.bar_duration,
                    bar_count=piece.bar_count,
                ),
                closing=closing,
                is_final=index == final_index,
                boundary_score=scores[index],
            )
        )

    return tuple(cadences)


def _boundary_score(
    slots: tuple[HarmonicSlot, ...],
    index: int,
    *,
    config: CadenceDetectionConfig,
    bar_duration: Fraction,
) -> float:
    slot = slots[index]
    authentic_arrival = (
        slot.tonic_triad_overlap
        if index > 0
        and slots[index - 1].function is HarmonicFunction.DOMINANT
        and slot.function is HarmonicFunction.TONIC
        else 0.0
    )
    half_arrival = min(1.0, slot.dwell) if slot.function is HarmonicFunction.DOMINANT else 0.0
    harmonic_arrival = max(authentic_arrival, half_arrival)
    rhythmic_stop = min(1.0, slot.dwell)
    bar_aligned = 1.0 if (slot.end / bar_duration).denominator == 1 else 0.0
    return (
        config.metrical_weight * slot.metrical_weight
        + config.harmonic_arrival_weight * harmonic_arrival
        + config.rhythmic_stop_weight * rhythmic_stop
        + config.bar_alignment_weight * bar_aligned
    )


def _select_arrivals(
    scores: list[float],
    *,
    threshold: float,
    separation: int,
    mandatory_index: int,
) -> tuple[int, ...]:
    selected = [mandatory_index]
    ordered = sorted(
        (index for index in range(len(scores)) if scores[index] >= threshold and index != mandatory_index),
        key=lambda index: (-scores[index], index),
    )
    for index in ordered:
        if all(abs(index - chosen) > separation for chosen in selected):
            selected.append(index)

    return tuple(sorted(selected))


def _closing_pattern(
    slots: tuple[HarmonicSlot, ...],
    index: int,
    *,
    maximum_closing_slots: int,
) -> ClosingPattern | None:
    collected: list[HarmonicFunction] = []
    for offset in range(maximum_closing_slots):
        position = index - offset
        if position < 0:
            break

        function = slots[position].function
        if function is None:
            break

        if collected and collected[-1] is function:
            continue

        collected.append(function)
        if function is HarmonicFunction.PREDOMINANT:
            break

    if not collected:
        return None

    return ClosingPattern(tuple(reversed(collected)))


def _slot_end_bar(
    slot: HarmonicSlot,
    *,
    bar_duration: Fraction,
    bar_count: int,
) -> int:
    bars = slot.end / bar_duration
    whole = int(bars)
    return min(bar_count, whole if bars == whole else whole + 1)
