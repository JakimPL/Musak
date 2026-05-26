from dataclasses import dataclass


@dataclass(frozen=True)
class FigureFrequencyMass:
    common: float
    rare: float
    novel: float
