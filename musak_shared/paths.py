from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

STATIC_DIR: Path = PROJECT_ROOT / "static"
TEMPLATES_DIR: Path = PROJECT_ROOT / "templates"
