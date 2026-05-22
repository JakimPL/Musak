import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from musak_model.data.config import SegmentationConfig
from musak_model.processing.config import TokenizationProcessingConfig
from musak_model.processing.snapshot import TokenizerSnapshot

TOKENIZATION_STATE_VERSION: Final[int] = 1
TOKENIZATION_STATE_HEADER: Final[str] = "header"
TOKENIZATION_SOURCE_COMPLETED: Final[str] = "source_completed"


@dataclass(frozen=True)
class TokenizationResumeState:
    completed_source_ids: frozenset[str]
    encoded_line_count: int
    manifest_row_count: int
    encoded_count: int
    state_key_matches: bool


def empty_resume_state() -> TokenizationResumeState:
    return TokenizationResumeState(
        completed_source_ids=frozenset(),
        encoded_line_count=0,
        manifest_row_count=0,
        encoded_count=0,
        state_key_matches=True,
    )


def tokenization_state_key(
    *,
    snapshot: TokenizerSnapshot,
    segmentation_config: SegmentationConfig,
    tokenization_processing_config: TokenizationProcessingConfig,
    difficulty_labels: dict[str, int | None] | None,
) -> str:
    payload = {
        "tokenizer_hash": snapshot.tokenizer_hash,
        "segmentation": segmentation_config.model_dump(mode="json"),
        "tokenization_processing": tokenization_processing_config.model_dump(mode="json"),
        "difficulty_labels": difficulty_labels or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_tokenization_resume_state(path: Path, *, state_key: str) -> TokenizationResumeState:
    if not path.exists():
        return empty_resume_state()

    completed_source_ids: set[str] = set()
    encoded_line_count = 0
    manifest_row_count = 0
    encoded_count = 0
    state_key_matches = False
    for event in _tokenization_state_events(path):
        if event.get("state_key") != state_key:
            return TokenizationResumeState(
                completed_source_ids=frozenset(),
                encoded_line_count=0,
                manifest_row_count=0,
                encoded_count=0,
                state_key_matches=False,
            )

        if event["type"] == TOKENIZATION_STATE_HEADER:
            state_key_matches = event.get("version") == TOKENIZATION_STATE_VERSION
            continue

        if event["type"] == TOKENIZATION_SOURCE_COMPLETED:
            completed_source_ids.add(str(event["source_id"]))
            encoded_line_count = _event_int(event, "encoded_line_count")
            manifest_row_count = _event_int(event, "manifest_row_count")
            encoded_count = _event_int(event, "encoded_count")

    return TokenizationResumeState(
        completed_source_ids=frozenset(completed_source_ids),
        encoded_line_count=encoded_line_count,
        manifest_row_count=manifest_row_count,
        encoded_count=encoded_count,
        state_key_matches=state_key_matches,
    )


def append_tokenization_state_header(path: Path, *, state_key: str) -> None:
    append_tokenization_state_event(
        path,
        {
            "type": TOKENIZATION_STATE_HEADER,
            "version": TOKENIZATION_STATE_VERSION,
            "state_key": state_key,
        },
    )


def append_source_completed_event(
    path: Path,
    *,
    state_key: str,
    source_id: str,
    encoded_line_count: int,
    manifest_row_count: int,
    encoded_count: int,
) -> None:
    append_tokenization_state_event(
        path,
        {
            "type": TOKENIZATION_SOURCE_COMPLETED,
            "state_key": state_key,
            "source_id": source_id,
            "encoded_line_count": encoded_line_count,
            "manifest_row_count": manifest_row_count,
            "encoded_count": encoded_count,
        },
    )


def append_tokenization_state_event(path: Path, event: dict[str, int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, sort_keys=True))
        file.write("\n")


def resume_state_outputs_missing(
    *,
    resume_state: TokenizationResumeState,
    encoded_jsonl_path: Path,
    encoded_manifest_path: Path,
) -> bool:
    if resume_state.encoded_line_count == 0 and resume_state.manifest_row_count == 0:
        return False

    return not (encoded_jsonl_path.exists() and encoded_manifest_path.exists())


def _tokenization_state_events(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() != "":
            events.append(json.loads(line))

    return events


def _event_int(event: dict[str, object], key: str) -> int:
    value = event[key]
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        return int(value)

    raise ValueError(f"tokenization state event field {key} must be an integer")
