from pathlib import Path

from musak_model.processing.paths import ENCODED_JSONL_NAME, TOKENIZER_SNAPSHOT_NAME


def encoded_run_directories(encoded_root: Path) -> list[Path]:
    if not encoded_root.exists():
        return []

    return sorted(
        path
        for path in encoded_root.iterdir()
        if path.is_dir() and (path / ENCODED_JSONL_NAME).is_file() and (path / TOKENIZER_SNAPSHOT_NAME).is_file()
    )


def resolve_encoded_directory(
    *,
    data_directory: Path | None,
    processed_root: Path,
    encoded_directory: Path | None,
) -> Path:
    if encoded_directory is not None:
        return encoded_directory

    if data_directory is None:
        raise ValueError("--data-dir is required when --encoded-directory is omitted")

    encoded_root = processed_root / data_directory.name / "encoded"
    encoded_directories = encoded_run_directories(encoded_root)
    if not encoded_directories:
        raise FileNotFoundError(f"No encoded runs found in {encoded_root}")

    if len(encoded_directories) > 1:
        raise ValueError(f"Multiple encoded runs found in {encoded_root}; pass --encoded-directory explicitly")

    return encoded_directories[0]
