import numpy as np
from numpy.typing import NDArray


def trend_and_residual(
    centered: NDArray[np.float64], *, arch_basis_count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    length = centered.size
    trend = np.zeros(length, dtype=np.float64)
    for index in range(1, min(arch_basis_count, length - 1) + 1):
        basis = mid_cell_dct_basis(index, length)
        trend += dct_projection_coefficient(centered, basis) * basis

    return trend, centered - trend


def mid_cell_dct_basis(index: int, length: int) -> NDArray[np.float64]:
    steps = np.arange(length)
    return np.cos(np.pi * index * (steps + 0.5) / length)


def dct_projection_coefficient(values: NDArray[np.float64], basis: NDArray[np.float64]) -> float:
    return 2.0 / values.size * float(values @ basis)
