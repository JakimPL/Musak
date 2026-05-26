from pathlib import Path

from musak_model.paths import ROOT_DIR


def sqlite_tracking_uri(database_path: Path) -> str:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database_path.as_posix()}"


def local_mlflow_tracking_uri(
    *,
    database_path: Path,
    tracking_root: Path,
) -> str:
    if database_path.is_absolute():
        try:
            database_path = database_path.relative_to(ROOT_DIR)
        except ValueError:
            return sqlite_tracking_uri(database_path)

    return sqlite_tracking_uri(tracking_root / database_path)
