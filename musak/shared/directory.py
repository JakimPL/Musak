import pathlib
import uuid
from typing import Final

TEMP_DIRECTORY: Final[str] = "temp"


def create_directory() -> tuple[str, pathlib.Path]:
    uuid64 = str(uuid.uuid4())
    temp_dir = pathlib.Path(TEMP_DIRECTORY)
    temp_dir.mkdir(exist_ok=True)
    directory = temp_dir / uuid64
    directory.mkdir()
    return uuid64, directory
