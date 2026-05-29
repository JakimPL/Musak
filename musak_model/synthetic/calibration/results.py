import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from musak_model.synthetic.calibration.schema import SweepResult

_COLUMNS: Final = (
    "lambda_curve",
    "lambda_harm",
    "lambda_accent",
    "distribution_groups",
    "mean_total_variation_distance",
)


def write_sweep_results(results: Sequence[SweepResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(_COLUMNS)
        for result in results:
            writer.writerow(
                [
                    result.lambda_curve,
                    result.lambda_harm,
                    result.lambda_accent,
                    result.distribution_groups,
                    "" if result.mean_total_variation_distance is None else result.mean_total_variation_distance,
                ]
            )
