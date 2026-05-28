import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


def band_limited_random(
    *,
    length: int,
    basis_count: int,
    amplitude: float,
    decay: float,
    rng: Generator,
) -> NDArray[np.float64]:
    if length <= 0:
        raise ValueError("length must be positive")

    if basis_count <= 0:
        raise ValueError("basis_count must be positive")

    basis_indices = np.arange(1, basis_count + 1)
    standard_deviations = amplitude / basis_indices.astype(np.float64) ** decay
    coefficients = rng.normal(loc=0.0, scale=standard_deviations)
    steps = np.arange(length)
    basis_matrix = np.cos(np.pi * np.outer(basis_indices, (steps + 0.5) / length))
    return coefficients @ basis_matrix
