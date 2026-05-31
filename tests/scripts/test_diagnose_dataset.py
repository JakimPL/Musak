from pathlib import Path

from scripts import diagnose_dataset


def test_dataset_name_uses_data_directory_when_available() -> None:
    assert diagnose_dataset.dataset_name(data_directory=Path("data/PDMX"), encoded_directory=None) == "PDMX"


def test_processed_directory_uses_configured_processed_root() -> None:
    assert diagnose_dataset.processed_directory(
        data_directory=Path("data/PDMX"),
        processed_root=Path("custom/processed"),
        encoded_directory=None,
    ) == Path("custom/processed/PDMX")


def test_dataset_name_can_be_inferred_from_encoded_directory() -> None:
    encoded_directory = Path("artifacts/processed/PDMX/encoded/hash")

    assert diagnose_dataset.dataset_name(data_directory=None, encoded_directory=encoded_directory) == "PDMX"
    assert diagnose_dataset.processed_directory(
        data_directory=None,
        processed_root=Path("ignored"),
        encoded_directory=encoded_directory,
    ) == Path("artifacts/processed/PDMX")
