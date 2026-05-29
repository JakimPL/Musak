from collections import Counter
from dataclasses import dataclass

from musak_model.n_grams.profile.artifacts import FigureArtifactPaths
from musak_model.n_grams.profile.schema import FigureProfile


@dataclass(frozen=True)
class SplitFigureArtifacts:
    profile: FigureProfile
    paths: FigureArtifactPaths


@dataclass(frozen=True)
class FigureCountGroup:
    key: tuple[str, str, int]
    counts: Counter[str]
