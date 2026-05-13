from pathlib import Path
from typing import Final

_PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
CONFIGS_DIR: Final[Path] = _PACKAGE_DIR / "configs"
