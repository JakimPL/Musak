from collections.abc import Sequence
from pathlib import Path
from typing import Final

from musak_model.synthetic.calibration.schema import SweepResult
from musak_shared.files import write_csv_rows

_COLUMNS: Final = (
    "lambda_curve",
    "lambda_harmonic",
    "lambda_accent",
    "distribution_groups",
    "mean_total_variation_distance",
)


def write_sweep_results(results: Sequence[SweepResult], path: Path) -> None:
    write_csv_rows(
        path,
        columns=_COLUMNS,
        rows=(
            {
                "lambda_curve": result.lambda_curve,
                "lambda_harmonic": result.lambda_harmonic,
                "lambda_accent": result.lambda_accent,
                "distribution_groups": result.distribution_groups,
                "mean_total_variation_distance": (
                    "" if result.mean_total_variation_distance is None else result.mean_total_variation_distance
                ),
            }
            for result in results
        ),
    )
