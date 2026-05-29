from pathlib import Path
from typing import Final

PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
ROOT_DIR: Final[Path] = PACKAGE_DIR.parent

CONFIGS_DIR: Final[Path] = PACKAGE_DIR / "configs"
MODEL_CONFIG_DIR: Final[Path] = CONFIGS_DIR / "model"
CONDITIONING_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "conditioning" / "conditioning.yml"
N_GRAM_ANALYSIS_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "analysis" / "n_grams.yml"
CHORD_VOCABULARY_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "generation" / "chords.yml"
CHORD_DECODING_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "generation" / "chord_decoding.yml"
REGISTER_CURVE_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "generation" / "register_curve.yml"
ACCENT_FIELD_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "generation" / "accent_field.yml"
HAND_COUPLING_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "generation" / "hand_coupling.yml"
INGESTION_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "training" / "ingestion.yml"
PROCESSING_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "data" / "processing.yml"
SEGMENTATION_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "data" / "segmentation.yml"
TOKENIZATION_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "tokens" / "tokenization.yml"
PRETRAINING_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "training" / "pretraining.yml"
FINETUNING_CONFIG_PATH: Final[Path] = CONFIGS_DIR / "training" / "finetuning.yml"

DEFAULT_DATA_DIR: Final[Path] = ROOT_DIR / "data"
ARTIFACTS_DIR: Final[Path] = ROOT_DIR / "artifacts"
DEFAULT_ANALYSIS_DIR: Final[Path] = ARTIFACTS_DIR / "analysis"
DEFAULT_MLFLOW_DIR: Final[Path] = ARTIFACTS_DIR / "mlflow"
DEFAULT_MLFLOW_DB_PATH: Final[Path] = DEFAULT_MLFLOW_DIR / "mlflow.db"
DEFAULT_PROCESSED_ROOT: Final[Path] = ARTIFACTS_DIR / "processed"
DEFAULT_PROFILE_OUTPUT_DIR: Final[Path] = ARTIFACTS_DIR / "profiles"
DEFAULT_PROCESSING_PROFILE_OUTPUT_DIR: Final[Path] = DEFAULT_PROFILE_OUTPUT_DIR / "processing"
DEFAULT_PARSING_PROFILE_OUTPUT_DIR: Final[Path] = DEFAULT_PROCESSING_PROFILE_OUTPUT_DIR / "parse"
DEFAULT_TOKENIZATION_PROFILE_OUTPUT_DIR: Final[Path] = DEFAULT_PROCESSING_PROFILE_OUTPUT_DIR / "tokenize"
DEFAULT_COMBINED_PROCESSING_PROFILE_OUTPUT_DIR: Final[Path] = DEFAULT_PROCESSING_PROFILE_OUTPUT_DIR / "process"
DEFAULT_CHECKPOINT_DIR: Final[Path] = ARTIFACTS_DIR / "checkpoints"
DEFAULT_PRETRAINING_CHECKPOINT_DIR: Final[Path] = DEFAULT_CHECKPOINT_DIR / "pretraining"
DEFAULT_FINETUNING_CHECKPOINT_DIR: Final[Path] = DEFAULT_CHECKPOINT_DIR / "finetuning"
DEFAULT_TRAINING_FIGURE_DIR: Final[Path] = ARTIFACTS_DIR / "figure-splits"
