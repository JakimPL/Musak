from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

PARSED_MANIFEST_NAME: Final[str] = "parsed.csv"
ENCODED_MANIFEST_NAME: Final[str] = "encoded.csv"
ENCODED_JSONL_NAME: Final[str] = "data-00000.jsonl"
TOKENIZER_SNAPSHOT_NAME: Final[str] = "tokenizer.json"


@dataclass(frozen=True)
class ProcessedDatasetPaths:
    root: Path

    @classmethod
    def from_dataset_root(
        cls,
        *,
        processed_root: Path,
        dataset_root: Path,
    ) -> ProcessedDatasetPaths:
        return cls(root=processed_root / dataset_root.name)

    @property
    def parsed_dir(self) -> Path:
        return self.root / "parsed"

    @property
    def parsed_manifest_path(self) -> Path:
        return self.root / PARSED_MANIFEST_NAME

    def parsed_score_path(self, source_id_value: str) -> Path:
        if source_id_value == "":
            raise ValueError("source_id_value must not be empty")

        return self.parsed_dir / source_id_value[0] / f"{source_id_value}.json"

    def encoded_dir(self, tokenizer_hash_value: str) -> Path:
        return self.root / "encoded" / tokenizer_hash_value

    def encoded_manifest_path(self, tokenizer_hash_value: str) -> Path:
        return self.encoded_dir(tokenizer_hash_value) / ENCODED_MANIFEST_NAME

    def encoded_jsonl_path(self, tokenizer_hash_value: str) -> Path:
        return self.encoded_dir(tokenizer_hash_value) / ENCODED_JSONL_NAME

    def tokenizer_snapshot_path(self, tokenizer_hash_value: str) -> Path:
        return self.encoded_dir(tokenizer_hash_value) / TOKENIZER_SNAPSHOT_NAME
