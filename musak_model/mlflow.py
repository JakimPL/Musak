import os
from pathlib import Path
from typing import Final

from musak_model.paths import DEFAULT_MLFLOW_DB_PATH, ROOT_DIRECTORY

_MLFLOW_TRACKING_URI_ENV: Final[str] = "MLFLOW_TRACKING_URI"
MLFLOW_RUN_ID_FILE_NAME: Final[str] = "mlflow_run_id.txt"

type ParamValue = str | int | float | bool


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
            database_path = database_path.relative_to(ROOT_DIRECTORY)
        except ValueError:
            return sqlite_tracking_uri(database_path)

    return sqlite_tracking_uri(tracking_root / database_path)


def resolve_tracking_uri(*, configured_uri: str | None, tracking_root: Path) -> str:
    if configured_uri is not None:
        return configured_uri

    environment_uri = os.getenv(_MLFLOW_TRACKING_URI_ENV)
    if environment_uri:
        return environment_uri

    return local_mlflow_tracking_uri(
        database_path=DEFAULT_MLFLOW_DB_PATH,
        tracking_root=tracking_root,
    )


def mlflow_run_id_path(checkpoint_directory: Path) -> Path:
    return checkpoint_directory / MLFLOW_RUN_ID_FILE_NAME


def write_mlflow_run_id(*, checkpoint_directory: Path, run_id: str) -> None:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    mlflow_run_id_path(checkpoint_directory).write_text(f"{run_id}\n", encoding="utf-8")


def read_mlflow_run_id(checkpoint_directory: Path) -> str | None:
    run_id_path = mlflow_run_id_path(checkpoint_directory)
    if not run_id_path.exists():
        return None

    run_id = run_id_path.read_text(encoding="utf-8").strip()
    if not run_id:
        return None

    return run_id


def flatten_params(values: dict[str, object], *, prefix: str = "") -> dict[str, ParamValue]:
    flattened: dict[str, ParamValue] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flattened.update(flatten_params(value, prefix=f"{name}."))
        elif isinstance(value, (str, int, float, bool)):
            flattened[name] = value
        elif value is None:
            flattened[name] = "null"
        else:
            flattened[name] = str(value)

    return flattened
