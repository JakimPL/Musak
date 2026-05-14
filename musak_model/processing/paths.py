from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
        return self.root / "parsed.csv"

    def parsed_score_path(self, source_id_value: str) -> Path:
        if source_id_value == "":
            raise ValueError("source_id_value must not be empty")

        return self.parsed_dir / source_id_value[0] / f"{source_id_value}.json"

    def encoded_dir(self, tokenizer_hash_value: str) -> Path:
        return self.root / "encoded" / tokenizer_hash_value

    def encoded_manifest_path(self, tokenizer_hash_value: str) -> Path:
        return self.encoded_dir(tokenizer_hash_value) / "encoded.csv"

    def encoded_jsonl_path(self, tokenizer_hash_value: str) -> Path:
        return self.encoded_dir(tokenizer_hash_value) / "data-00000.jsonl"

    def tokenizer_snapshot_path(self, tokenizer_hash_value: str) -> Path:
        return self.encoded_dir(tokenizer_hash_value) / "tokenizer.json"
