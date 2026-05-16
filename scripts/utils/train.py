from __future__ import annotations

import argparse
from enum import StrEnum
from pathlib import Path

import torch

from musak_model.data.config import load_difficulty_labels, load_segmentation_config
from musak_model.paths import (
    CONDITIONING_CONFIG_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_MLFLOW_DIR,
    DEFAULT_STAGE_ONE_CHECKPOINT_DIR,
    DEFAULT_STAGE_TWO_CHECKPOINT_DIR,
    INGESTION_CONFIG_PATH,
    SEGMENTATION_CONFIG_PATH,
    STAGE_TWO_TRAINING_CONFIG_PATH,
    TOKENIZATION_CONFIG_PATH,
    TRAINING_CONFIG_PATH,
)
from musak_model.tokens.config import TokenizationConfig
from musak_model.training.config import StageTwoTrainingConfig, TrainingConfig
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.stage_two import train_stage_two
from musak_model.training.trainer import TrainingResult, train_stage_one


class TrainingStage(StrEnum):
    STAGE_ONE = "stage-one"
    STAGE_TWO = "stage-two"


class TrainHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def run_training(stage: TrainingStage) -> None:
    args = parse_training_args(stage)
    ingestion_config = build_ingestion_config(args)
    segmentation_config = load_segmentation_config(
        args.segmentation_config,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
    )
    tokenization_config = TokenizationConfig.load(args.tokenization_config)
    match stage:
        case TrainingStage.STAGE_ONE:
            result = train_stage_one(
                args.data_dir,
                ingestion_config=ingestion_config,
                segmentation_config=segmentation_config,
                training_config=build_stage_one_training_config(args),
                tokenization_config=tokenization_config,
                conditioning_config_path=args.conditioning_config,
            )
        case TrainingStage.STAGE_TWO:
            result = train_stage_two(
                args.data_dir,
                ingestion_config=ingestion_config,
                segmentation_config=segmentation_config,
                training_config=build_stage_two_training_config(args),
                tokenization_config=tokenization_config,
                conditioning_config_path=args.conditioning_config,
            )

    print_training_result(result)


def parse_training_args(stage: TrainingStage) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=_description(stage),
        epilog=_epilog(stage),
        formatter_class=TrainHelpFormatter,
    )
    add_common_training_arguments(parser, stage=stage)
    return parser.parse_args()


def add_common_training_arguments(parser: argparse.ArgumentParser, *, stage: TrainingStage) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Dataset root. With --processed-dir, artifacts are resolved as <processed-dir>/<data-dir.name>/.",
    )
    parser.add_argument("--ingestion-config", type=Path, default=INGESTION_CONFIG_PATH, help="Ingestion YAML config.")
    parser.add_argument(
        "--segmentation-config",
        type=Path,
        default=SEGMENTATION_CONFIG_PATH,
        help="Segmentation YAML config.",
    )
    parser.add_argument(
        "--tokenization-config",
        type=Path,
        default=TOKENIZATION_CONFIG_PATH,
        help="Tokenization YAML config used for raw/parsed tokenization and encoded artifact hash matching.",
    )
    parser.add_argument(
        "--conditioning-config",
        type=Path,
        default=CONDITIONING_CONFIG_PATH,
        help="Conditioning YAML config.",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=_training_config_path(stage),
        help="Training YAML config.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for training checkpoints.",
    )
    if stage == TrainingStage.STAGE_TWO:
        parser.add_argument(
            "--stage-one-checkpoint",
            type=Path,
            default=None,
            help="Stage-one checkpoint whose model weights initialize stage-two fine-tuning.",
        )
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--mlflow-dir", type=Path, default=DEFAULT_MLFLOW_DIR, help="Local MLflow tracking directory.")
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow tracking.")
    parser.add_argument("--mlflow-experiment-name", type=str, default=None, help="Override MLflow experiment name.")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="Override MLflow run name.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None, help="Training device.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count.")
    parser.add_argument("--validation-fraction", type=float, default=None, help="Override validation split fraction.")
    parser.add_argument("--split-seed", type=int, default=None, help="Override deterministic split seed.")
    parser.add_argument("--window-bars", type=int, default=None, help="Override segment window size in bars.")
    parser.add_argument("--stride-bars", type=int, default=None, help="Override segment stride in bars.")
    parser.add_argument("--difficulty-labels", type=Path, default=None, help="Optional YAML difficulty-label mapping.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Processed artifact root. Training resolves artifacts under <processed-dir>/<data-dir.name>/.",
    )
    if stage == TrainingStage.STAGE_ONE:
        parser.add_argument(
            "--use-conditioning",
            action="store_true",
            help="Enable conditioning inputs during training.",
        )


def build_ingestion_config(args: argparse.Namespace) -> IngestionConfig:
    config = IngestionConfig.load(args.ingestion_config)
    difficulty_labels = load_difficulty_labels(args.difficulty_labels)
    return config.model_copy(
        update={
            key: value
            for key, value in {
                "validation_fraction": args.validation_fraction,
                "split_seed": args.split_seed,
                "difficulty_labels": difficulty_labels if difficulty_labels is not None else config.difficulty_labels,
                "processed_root": args.processed_dir if args.processed_dir is not None else config.processed_root,
            }.items()
            if value is not None
        }
    )


def build_stage_one_training_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig.load(args.training_config)
    updates = common_training_updates(
        args,
        config=config,
        default_checkpoint_dir=DEFAULT_STAGE_ONE_CHECKPOINT_DIR,
    )
    updates["use_conditioning"] = args.use_conditioning or config.use_conditioning
    return config.model_copy(update={key: value for key, value in updates.items() if value is not None})


def build_stage_two_training_config(args: argparse.Namespace) -> StageTwoTrainingConfig:
    config = StageTwoTrainingConfig.load(args.training_config)
    updates = common_training_updates(
        args,
        config=config,
        default_checkpoint_dir=DEFAULT_STAGE_TWO_CHECKPOINT_DIR,
    )
    updates["stage_one_checkpoint"] = (
        args.stage_one_checkpoint if args.stage_one_checkpoint is not None else config.stage_one_checkpoint
    )
    updates["use_conditioning"] = True
    updates["use_structural_conditioning"] = True
    return config.model_copy(update={key: value for key, value in updates.items() if value is not None})


def common_training_updates(
    args: argparse.Namespace,
    *,
    config: TrainingConfig,
    default_checkpoint_dir: Path,
) -> dict[str, int | float | bool | str | Path | None]:
    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "checkpoint_dir": args.checkpoint_dir or config.checkpoint_dir or default_checkpoint_dir,
        "resume_checkpoint": args.resume_checkpoint if args.resume_checkpoint is not None else config.resume_checkpoint,
        "device": resolve_device(args.device or config.device),
        "enable_mlflow": not args.disable_mlflow and config.enable_mlflow,
        "mlflow_experiment_name": args.mlflow_experiment_name or config.mlflow_experiment_name,
        "mlflow_run_name": args.mlflow_run_name if args.mlflow_run_name is not None else config.mlflow_run_name,
        "mlflow_tracking_uri": str(args.mlflow_dir),
    }


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def print_training_result(result: TrainingResult) -> None:
    metrics = result.metrics
    if metrics:
        last_metric = metrics[-1]
        print(
            f"finished epochs={len(metrics)} "
            f"last_train_loss={last_metric.train_loss:.6f} "
            f"last_validation_loss={last_metric.validation_loss}"
        )
    print(f"latest_checkpoint={result.latest_checkpoint_path}")
    print(f"best_checkpoint={result.best_checkpoint_path}")
    print(f"invalid_files={len(result.invalid_files)}")


def _description(stage: TrainingStage) -> str:
    match stage:
        case TrainingStage.STAGE_ONE:
            return (
                "Train the Musak Stage 1 autoregressive model. The data directory is the dataset root; "
                "when --processed-dir is set, training looks for reusable artifacts under "
                "<processed-dir>/<data-dir.name>/ before falling back to raw MusicXML."
            )
        case TrainingStage.STAGE_TWO:
            return (
                "Fine-tune the Musak Stage 2 constrained autoregressive model from a Stage 1 checkpoint. "
                "The data directory follows the same artifact lookup rules as Stage 1."
            )


def _epilog(stage: TrainingStage) -> str:
    executable = _executable(stage)
    stage_two_extra = ""
    if stage == TrainingStage.STAGE_TWO:
        stage_two_extra = " --stage-one-checkpoint checkpoints/stage_one/best.pt"
    return (
        "Examples:\n"
        f"  uv run python {executable} --data-dir data/PDMX --processed-dir processed{stage_two_extra}\n"
        f"  uv run python {executable} --data-dir data/PDMX --processed-dir processed "
        "--tokenization-config musak_model/configs/tokens/tokenization.yml"
        f"{stage_two_extra}\n\n"
        "Artifact lookup:\n"
        "  Encoded artifacts are reused only from encoded/<tokenizer-hash>/ when tokenizer.json matches "
        "the active tokenization config.\n"
        "  If matching encoded artifacts are unavailable, parsed JSON artifacts are used when present.\n"
        "  If no usable processed artifacts exist, MusicXML files are parsed on the fly."
    )


def _executable(stage: TrainingStage) -> str:
    match stage:
        case TrainingStage.STAGE_ONE:
            return "scripts/train_stage_one.py"
        case TrainingStage.STAGE_TWO:
            return "scripts/train_stage_two.py"


def _training_config_path(stage: TrainingStage) -> Path:
    match stage:
        case TrainingStage.STAGE_ONE:
            return TRAINING_CONFIG_PATH
        case TrainingStage.STAGE_TWO:
            return STAGE_TWO_TRAINING_CONFIG_PATH
