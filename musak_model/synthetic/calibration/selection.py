from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from musak_model.synthetic.calibration.config import CalibrationConfig
from musak_model.synthetic.calibration.schema import SweepResult
from musak_model.synthetic.substitution import SubstitutionConfig

_DIRECTIONS: Final = ("lambda_curve", "lambda_harm", "lambda_accent")


@dataclass(frozen=True)
class TiltSelection:
    lambda_curve: float
    lambda_harm: float
    lambda_accent: float
    threshold_met: bool


def select_tilts(
    results: Sequence[SweepResult],
    *,
    lambda_curve: tuple[float, ...],
    lambda_harm: tuple[float, ...],
    lambda_accent: tuple[float, ...],
    threshold: float,
) -> TiltSelection:
    grids = {"lambda_curve": lambda_curve, "lambda_harm": lambda_harm, "lambda_accent": lambda_accent}
    baselines = {direction: _baseline(grids[direction]) for direction in _DIRECTIONS}
    selected: dict[str, float] = {}
    threshold_met = True
    for direction in _DIRECTIONS:
        value, met = _select_direction(results, direction=direction, baselines=baselines, threshold=threshold)
        selected[direction] = value
        threshold_met = threshold_met and met

    return TiltSelection(
        lambda_curve=selected["lambda_curve"],
        lambda_harm=selected["lambda_harm"],
        lambda_accent=selected["lambda_accent"],
        threshold_met=threshold_met,
    )


def selected_substitution_config(selection: TiltSelection, config: CalibrationConfig) -> SubstitutionConfig:
    return SubstitutionConfig(
        lambda_curve=selection.lambda_curve,
        lambda_harm=selection.lambda_harm,
        lambda_accent=selection.lambda_accent,
        commonness_bias=config.commonness_bias,
        max_resample_retries=config.max_resample_retries,
        monophonic=False,
    )


def write_selected_substitution_config(
    selection: TiltSelection,
    config: CalibrationConfig,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(selected_substitution_config(selection, config).model_dump_json(indent=2), encoding="utf-8")


def _baseline(values: tuple[float, ...]) -> float:
    return 0.0 if 0.0 in values else min(values)


def _select_direction(
    results: Sequence[SweepResult],
    *,
    direction: str,
    baselines: dict[str, float],
    threshold: float,
) -> tuple[float, bool]:
    others = [other for other in _DIRECTIONS if other != direction]
    scored: list[tuple[float, float]] = []
    for result in results:
        distance = result.mean_total_variation_distance
        if distance is None or any(getattr(result, other) != baselines[other] for other in others):
            continue

        scored.append((float(getattr(result, direction)), distance))

    if not scored:
        return baselines[direction], False

    under_threshold = [value for value, distance in scored if distance <= threshold]
    if under_threshold:
        return max(under_threshold), True

    return min(scored, key=lambda item: item[1])[0], False
