from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

_HASH_LENGTH: Final[int] = 24


def source_id(
    source_path: Path,
    *,
    dataset_root: Path,
) -> str:
    relative_path = source_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    return _hash_text(relative_path)


def segment_id(
    source_id_value: str,
    *,
    window_start_bar: int,
    bar_count: int,
) -> str:
    return _hash_text(f"{source_id_value}:{window_start_bar}:{bar_count}")


def tokenizer_hash(snapshot: dict[str, Any]) -> str:
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return _hash_text(canonical)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:_HASH_LENGTH]
