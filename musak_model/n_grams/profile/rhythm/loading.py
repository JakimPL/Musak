from dataclasses import dataclass

from musak_model.n_grams.profile.artifacts import FigureArtifactPaths
from musak_model.n_grams.profile.rhythm.io import read_rhythm_counts, read_rhythm_profile
from musak_model.n_grams.profile.rhythm.schema import (
    RhythmArtifactPaths,
    RhythmCountCounter,
    RhythmProfile,
    rhythm_artifact_paths_for_figure_root,
)


@dataclass(frozen=True)
class RhythmProfileArtifacts:
    paths: RhythmArtifactPaths
    profile: RhythmProfile
    counts: RhythmCountCounter


def load_rhythm_profile_artifacts(paths: FigureArtifactPaths) -> RhythmProfileArtifacts | None:
    rhythm_paths = rhythm_artifact_paths_for_figure_root(paths.root_directory)
    missing_paths = tuple(path for path in (rhythm_paths.counts_path, rhythm_paths.profile_path) if not path.exists())
    if len(missing_paths) == 2:
        return None

    if missing_paths:
        raise FileNotFoundError(
            f"Rhythm profile artifacts are incomplete under {rhythm_paths.root_directory}; missing: "
            f"{', '.join(path.as_posix() for path in missing_paths)}"
        )

    return RhythmProfileArtifacts(
        paths=rhythm_paths,
        profile=read_rhythm_profile(rhythm_paths.profile_path),
        counts=read_rhythm_counts(rhythm_paths.counts_path),
    )
