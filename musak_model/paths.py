from pathlib import Path
from typing import Final

PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
ROOT_DIR: Final[Path] = PACKAGE_DIR.parent

CONFIGS_DIR: Final[Path] = PACKAGE_DIR / "configs"
MODEL_CONFIG_DIR: Final[Path] = CONFIGS_DIR / "model"
CONDITIONING_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "conditioning" / "conditioning.yml"
INGESTION_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "training" / "ingestion.yml"
SEGMENTATION_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "data" / "segmentation.yml"
TOKENIZATION_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "tokens" / "tokenization.yml"
TRAINING_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "training" / "stage_one.yml"
STAGE_TWO_TRAINING_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "training" / "stage_two.yml"

DEFAULT_DATA_DIR: Final[Path] = ROOT_DIR / "data"
DEFAULT_MLFLOW_DIR: Final[Path] = ROOT_DIR / "mlruns"
DEFAULT_PROCESSED_ROOT: Final[Path] = ROOT_DIR / "processed"
DEFAULT_STAGE_ONE_CHECKPOINT_DIR: Final[Path] = ROOT_DIR / "checkpoints" / "stage_one"
DEFAULT_STAGE_TWO_CHECKPOINT_DIR: Final[Path] = ROOT_DIR / "checkpoints" / "stage_two"
