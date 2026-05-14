from pathlib import Path
from typing import Final

_PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
CONFIGS_DIR: Final[Path] = _PACKAGE_DIR / "configs"
CONDITIONING_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "conditioning" / "conditioning.yml"
INGESTION_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "training" / "ingestion.yml"
MODEL_CONFIG_DIR: Final[Path] = CONFIGS_DIR / "model"
SEGMENTATION_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "data" / "segmentation.yml"
TOKENIZATION_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "tokens" / "tokenization.yml"
TRAINING_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "training" / "stage_one.yml"

DEFAULT_DATA_DIR: Final[Path] = Path("data")
DEFAULT_MLFLOW_DIR: Final[Path] = Path("mlruns")
DEFAULT_PROCESSED_ROOT: Final[Path] = Path("processed")
DEFAULT_STAGE_ONE_CHECKPOINT_DIR: Final[Path] = Path("checkpoints") / "stage_one"
