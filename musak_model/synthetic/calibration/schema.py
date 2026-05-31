from dataclasses import dataclass


@dataclass(frozen=True)
class SweepResult:
    lambda_curve: float
    lambda_harmonic: float
    lambda_accent: float
    distribution_groups: int
    mean_total_variation_distance: float | None
