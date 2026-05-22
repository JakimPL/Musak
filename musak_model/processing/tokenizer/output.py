import csv
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO

from pydantic import BaseModel

from musak_model.processing.manifest import ENCODED_MANIFEST_FIELDS, read_encoded_manifest


def append_jsonl_model(
    model: BaseModel,
    path: Path,
    *,
    line_index: int,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(model.model_dump_json())
        file.write("\n")

    return line_index


def append_encoded_manifest_rows(rows: list[dict[str, object]], path: Path) -> None:
    with EncodedManifestAppender(path) as appender:
        for row in rows:
            appender.append(row)


class EncodedManifestAppender:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> Self:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._path.exists()
        self._file = self._path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=[field.value for field in ENCODED_MANIFEST_FIELDS])
        if write_header:
            self._writer.writeheader()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._file is not None:
            self._file.close()

    def append(self, row: dict[str, object]) -> None:
        if self._writer is None:
            raise RuntimeError("encoded manifest appender is not open")

        self._writer.writerow(row)


def truncate_text_lines(path: Path, line_count: int) -> None:
    if not path.exists():
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("".join(f"{line}\n" for line in lines[:line_count]), encoding="utf-8")


def truncate_manifest_rows(path: Path, row_count: int) -> None:
    if not path.exists():
        return

    rows = read_encoded_manifest(path)[:row_count]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.value for field in ENCODED_MANIFEST_FIELDS])
        writer.writeheader()
        writer.writerows(rows)


def clear_tokenization_outputs(
    *,
    encoded_jsonl_path: Path,
    encoded_manifest_path: Path,
    tokenizer_snapshot_path: Path | None,
    state_path: Path,
) -> None:
    for path in (encoded_jsonl_path, encoded_manifest_path, tokenizer_snapshot_path, state_path):
        if path is not None:
            path.unlink(missing_ok=True)
