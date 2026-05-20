import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def encoded_samples_fingerprint(samples: Sequence[BaseModel]) -> str:
    records = [sample.model_dump(mode="json") for sample in samples]
    canonical = json.dumps(
        sorted(records, key=_encoded_sample_sort_key),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _encoded_sample_sort_key(record: dict[str, object]) -> tuple[str, int, int, str]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return ("", 0, 0, "")

    source_file = str(metadata.get("source_file", ""))
    window_start_bar = _int_value(metadata.get("window_start_bar"))
    bar_count = _int_value(metadata.get("bar_count"))
    token_ids = json.dumps(record.get("token_ids", []), separators=(",", ":"))
    return source_file, window_start_bar, bar_count, token_ids


def _int_value(value: object) -> int:
    if isinstance(value, int):
        return value

    return 0
