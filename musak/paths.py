from pathlib import Path

PACKAGE_ROOT: Path = Path(__file__).parent

INTERVALS_CONFIG: Path = PACKAGE_ROOT / "config" / "intervals.yml"
INVERSIONS_CONFIG: Path = PACKAGE_ROOT / "config" / "inversions.yml"
RHYTHM_CONFIG: Path = PACKAGE_ROOT / "config" / "rhythm.yml"
