import logging
from dataclasses import dataclass
from pathlib import Path

from musak_model.processing.parser import ParsedScoreArtifact

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DifficultyLabelStats:
    labeled: int
    explicit_unlabeled: int
    unspecified: int


def log_difficulty_label_stats(
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    *,
    dataset_root: Path,
    difficulty_labels: dict[str, int | None] | None,
) -> None:
    if difficulty_labels is None:
        return

    stats = difficulty_label_stats(parsed_scores, dataset_root=dataset_root, difficulty_labels=difficulty_labels)
    message = (
        f"Difficulty labels: labeled={stats.labeled} "
        f"explicit_unlabeled={stats.explicit_unlabeled} "
        f"unspecified={stats.unspecified}"
    )
    if stats.unspecified > 0:
        _LOGGER.warning(message)
    else:
        _LOGGER.info(message)


def difficulty_label_stats(
    parsed_scores: tuple[ParsedScoreArtifact, ...],
    *,
    dataset_root: Path,
    difficulty_labels: dict[str, int | None],
) -> DifficultyLabelStats:
    labeled = 0
    explicit_unlabeled = 0
    unspecified = 0
    for artifact in parsed_scores:
        relative_source_path = Path(artifact.source_path.resolve().relative_to(dataset_root.resolve()).as_posix())
        matching_key = _first_difficulty_label_key(relative_source_path, difficulty_labels)
        if matching_key is None:
            unspecified += 1
            continue

        if difficulty_labels[matching_key] is None:
            explicit_unlabeled += 1
        else:
            labeled += 1

    return DifficultyLabelStats(
        labeled=labeled,
        explicit_unlabeled=explicit_unlabeled,
        unspecified=unspecified,
    )


def _first_difficulty_label_key(path: Path, labels: dict[str, int | None]) -> str | None:
    for key in (path.as_posix(), path.name, path.stem):
        if key in labels:
            return key

    return None
