import csv
from pathlib import Path

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
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.value for field in ENCODED_MANIFEST_FIELDS])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


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
