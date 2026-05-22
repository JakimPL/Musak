import logging
from enum import StrEnum
from pathlib import Path

from musak_model.processing.io import load_parsed_score_json
from musak_model.processing.manifest import ParsedManifestField, ParsedManifestStatus, read_parsed_manifest
from musak_model.processing.parser.schema import ParsedScoreArtifact, ParsedScoreResult
from musak_model.processing.paths import ProcessedDatasetPaths

_LOGGER = logging.getLogger(__name__)


class ParsedManifestReuseIssue(StrEnum):
    UNSUPPORTED_STATUS = "unsupported_status"
    MISSING_PARSED_PATH = "missing_parsed_path"
    UNREADABLE_PARSED_SCORE = "unreadable_parsed_score"


def reusable_parsed_manifest_rows(
    *,
    paths: ProcessedDatasetPaths,
    overwrite: bool,
) -> dict[str, dict[str, str]]:
    if overwrite or not paths.parsed_manifest_path.exists():
        return {}

    rows = read_parsed_manifest(paths.parsed_manifest_path)
    _LOGGER.info("Loaded parsed manifest for resume: %s (%s row(s))", paths.parsed_manifest_path, len(rows))
    return {row[ParsedManifestField.SOURCE_ID]: row for row in rows if row.get(ParsedManifestField.SOURCE_ID, "")}


def reused_parsed_result(
    *,
    index: int,
    source_id_value: str,
    source_path: Path,
    paths: ProcessedDatasetPaths,
    row: dict[str, str] | None,
) -> tuple[ParsedScoreResult | None, ParsedManifestReuseIssue | None]:
    if row is None:
        return None, None

    parsed_path = paths.parsed_score_path(source_id_value)
    status = row[ParsedManifestField.STATUS]
    if status == ParsedManifestStatus.ERROR.value:
        return (
            ParsedScoreResult(
                index=index,
                source_id_value=source_id_value,
                source_path=source_path,
                parsed_path=parsed_path,
                row=dict(row),
                score=None,
            ),
            None,
        )

    if status != ParsedManifestStatus.SUCCESS.value:
        return None, ParsedManifestReuseIssue.UNSUPPORTED_STATUS

    parsed_path_text = row[ParsedManifestField.PARSED_PATH]
    if parsed_path_text == "":
        return None, ParsedManifestReuseIssue.MISSING_PARSED_PATH

    parsed_path = paths.root / parsed_path_text
    try:
        score = load_parsed_score_json(parsed_path)
    except (OSError, ValueError):
        return None, ParsedManifestReuseIssue.UNREADABLE_PARSED_SCORE

    return (
        ParsedScoreResult(
            index=index,
            source_id_value=source_id_value,
            source_path=source_path,
            parsed_path=parsed_path,
            row=dict(row),
            score=score,
        ),
        None,
    )


def load_parsed_score_artifacts(
    dataset_root: Path,
    *,
    processed_root: Path,
) -> tuple[ParsedScoreArtifact, ...]:
    paths = ProcessedDatasetPaths.from_dataset_root(processed_root=processed_root, dataset_root=dataset_root)
    if not paths.parsed_manifest_path.exists():
        raise FileNotFoundError(f"parsed manifest does not exist: {paths.parsed_manifest_path}")

    artifacts: list[ParsedScoreArtifact] = []
    for row in read_parsed_manifest(paths.parsed_manifest_path):
        if row[ParsedManifestField.STATUS] != ParsedManifestStatus.SUCCESS.value:
            continue

        artifacts.append(_artifact_from_success_row(row, dataset_root=dataset_root, paths=paths))

    return tuple(artifacts)


def _artifact_from_success_row(
    row: dict[str, str],
    *,
    dataset_root: Path,
    paths: ProcessedDatasetPaths,
) -> ParsedScoreArtifact:
    parsed_path_text = row[ParsedManifestField.PARSED_PATH]
    if parsed_path_text == "":
        raise ValueError(f"parsed manifest row is missing parsed_path: {row[ParsedManifestField.SOURCE_ID]}")

    parsed_path = paths.root / parsed_path_text
    if not parsed_path.exists():
        raise FileNotFoundError(f"parsed score artifact does not exist: {parsed_path}")

    return ParsedScoreArtifact(
        source_id_value=row[ParsedManifestField.SOURCE_ID],
        source_path=dataset_root / row[ParsedManifestField.SOURCE_PATH],
        parsed_path=parsed_path,
        score=load_parsed_score_json(parsed_path),
    )
