from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from musak_model.data.schema import ParsedScore
from musak_model.processing.snapshot import TokenizerSnapshot
from musak_shared.files import JSON_INDENT, line_count

if TYPE_CHECKING:
    from musak_model.training.ingestion.schema import EncodedExercise


def write_json_model(
    model: BaseModel,
    path: Path,
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=JSON_INDENT), encoding="utf-8")


def load_parsed_score_json(path: Path) -> ParsedScore:
    return ParsedScore.model_validate_json(path.read_text(encoding="utf-8"))


def load_tokenizer_snapshot_json(path: Path) -> TokenizerSnapshot:
    return TokenizerSnapshot.model_validate_json(path.read_text(encoding="utf-8"))


def append_jsonl(model: BaseModel, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    line_index = line_count(path)
    with path.open("a", encoding="utf-8") as file:
        file.write(model.model_dump_json())
        file.write("\n")

    return line_index


def load_encoded_jsonl(path: Path) -> list["EncodedExercise"]:
    from musak_model.training.ingestion.schema import EncodedExercise

    if not path.exists():
        return []

    return [
        EncodedExercise.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
