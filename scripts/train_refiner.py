from __future__ import annotations

import argparse
import logging
import shlex
from pathlib import Path

from musak_model.data.config import SegmentationMode, load_difficulty_labels, load_segmentation_config
from musak_model.mlflow import read_mlflow_run_id, sqlite_tracking_uri
from musak_model.paths import (
    DEFAULT_MLFLOW_DB_PATH,
    DEFAULT_RHYTHM_REFINER_CHECKPOINT_DIRECTORY,
    INGESTION_CONFIG_PATH,
    RHYTHM_REFINER_CONFIG_PATH,
    SEGMENTATION_CONFIG_PATH,
    TOKENIZATION_CONFIG_PATH,
)
from musak_model.rhythm_refiner.config import RhythmRefinerTrainingConfig
from musak_model.rhythm_refiner.training import RhythmRefinerTrainingResult, train_rhythm_refiner
from musak_model.tokens.config import TokenizationConfig
from musak_model.training.config import CheckpointConfig, MlflowConfig, OptimizationConfig, RuntimeConfig
from musak_model.training.ingestion.config import IngestionConfig
from scripts.utils.logger import DEFAULT_LOG_LEVEL, LOG_LEVEL_CHOICES, configure_logging
from scripts.utils.train import resolve_device

_LOGGER = logging.getLogger(__name__)


def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)
    result = _run(args)
    _print_result(result)


def _run(args: argparse.Namespace) -> RhythmRefinerTrainingResult:
    ingestion_config = _ingestion_config(args)
    segmentation_config = load_segmentation_config(
        args.segmentation_config,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
        mode=SegmentationMode.WHOLE_FILE if args.whole_file_segments else None,
    )
    tokenization_config = TokenizationConfig.load(args.tokenization_config)
    training_config = _training_config(args)
    _validate_checkpoint_safety(args, training_config)
    _log_start(args, training_config)
    try:
        return train_rhythm_refiner(
            args.data_dir,
            ingestion_config=ingestion_config,
            segmentation_config=segmentation_config,
            tokenization_config=tokenization_config,
            training_config=training_config,
            show_progress=not args.no_progress,
        )
    except KeyboardInterrupt as exception:
        _handle_keyboard_interrupt(args, training_config)
        raise SystemExit(130) from exception


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the isolated Musak rhythm-grid refiner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root. Processed artifacts are reused when available.",
    )
    parser.add_argument("--ingestion-config", type=Path, default=INGESTION_CONFIG_PATH, help="Ingestion YAML config.")
    parser.add_argument("--segmentation-config", type=Path, default=SEGMENTATION_CONFIG_PATH)
    parser.add_argument("--tokenization-config", type=Path, default=TOKENIZATION_CONFIG_PATH)
    parser.add_argument("--training-config", type=Path, default=RHYTHM_REFINER_CONFIG_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing refiner checkpoints.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--window-bars", type=int, default=None)
    parser.add_argument("--stride-bars", type=int, default=None)
    parser.add_argument("--whole-file-segments", action="store_true")
    parser.add_argument("--difficulty-labels", type=Path, default=None)
    parser.add_argument("--mlflow-db", type=Path, default=DEFAULT_MLFLOW_DB_PATH)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment-name", type=str, default=None)
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument("--mlflow-run-id", type=str, default=None)
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="Explicit MLflow tracking URI. Overrides --mlflow-db.",
    )
    parser.add_argument("--log-level", choices=LOG_LEVEL_CHOICES, default=DEFAULT_LOG_LEVEL)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _ingestion_config(args: argparse.Namespace) -> IngestionConfig:
    config = IngestionConfig.load(args.ingestion_config)
    difficulty_labels = load_difficulty_labels(args.difficulty_labels)
    return config.model_copy(
        update={
            key: value
            for key, value in {
                "validation_fraction": args.validation_fraction,
                "split_seed": args.split_seed,
                "difficulty_labels": difficulty_labels if difficulty_labels is not None else config.difficulty_labels,
            }.items()
            if value is not None
        }
    )


def _training_config(args: argparse.Namespace) -> RhythmRefinerTrainingConfig:
    config = RhythmRefinerTrainingConfig.load(args.training_config)
    return config.model_copy(
        update={
            "optimization": OptimizationConfig(
                epochs=args.epochs if args.epochs is not None else config.optimization.epochs,
                batch_size=args.batch_size if args.batch_size is not None else config.optimization.batch_size,
                learning_rate=(
                    args.learning_rate if args.learning_rate is not None else config.optimization.learning_rate
                ),
                weight_decay=args.weight_decay if args.weight_decay is not None else config.optimization.weight_decay,
            ),
            "runtime": RuntimeConfig(
                num_workers=args.num_workers if args.num_workers is not None else config.runtime.num_workers,
                device=resolve_device(args.device or config.runtime.device),
            ),
            "checkpoints": CheckpointConfig(
                checkpoint_directory=args.checkpoint_dir
                or config.checkpoints.checkpoint_directory
                or DEFAULT_RHYTHM_REFINER_CHECKPOINT_DIRECTORY,
                resume_checkpoint=(
                    args.resume_checkpoint
                    if args.resume_checkpoint is not None
                    else config.checkpoints.resume_checkpoint
                ),
                save_all_epochs=config.checkpoints.save_all_epochs,
            ),
            "mlflow": MlflowConfig(
                enable_mlflow=not args.disable_mlflow and config.mlflow.enable_mlflow,
                mlflow_experiment_name=args.mlflow_experiment_name or config.mlflow.mlflow_experiment_name,
                mlflow_run_name=(
                    args.mlflow_run_name if args.mlflow_run_name is not None else config.mlflow.mlflow_run_name
                ),
                mlflow_run_id=args.mlflow_run_id if args.mlflow_run_id is not None else config.mlflow.mlflow_run_id,
                mlflow_tracking_uri=args.mlflow_tracking_uri or sqlite_tracking_uri(args.mlflow_db),
            ),
        }
    )


def _validate_checkpoint_safety(args: argparse.Namespace, training_config: RhythmRefinerTrainingConfig) -> None:
    existing = tuple(
        path
        for path in (
            training_config.checkpoints.checkpoint_directory / "latest.pt",
            training_config.checkpoints.checkpoint_directory / "best.pt",
        )
        if path.exists()
    )
    if training_config.checkpoints.resume_checkpoint is not None or args.overwrite or not existing:
        return

    existing_list = ", ".join(str(path) for path in existing)
    raise FileExistsError(
        "Refiner checkpoint file(s) already exist. Pass --resume-checkpoint or --overwrite. "
        f"Existing files: {existing_list}"
    )


def _log_start(args: argparse.Namespace, training_config: RhythmRefinerTrainingConfig) -> None:
    _LOGGER.info("Starting rhythm refiner training")
    _LOGGER.info("Data directory: %s", args.data_dir)
    _LOGGER.info("Training config: %s", args.training_config)
    _LOGGER.info("Tokenization config: %s", args.tokenization_config)
    _LOGGER.info("Segmentation config: %s", args.segmentation_config)
    _LOGGER.info("Device: %s", training_config.runtime.device)
    _LOGGER.info("Epochs: %s", training_config.optimization.epochs)
    _LOGGER.info("Batch size: %s", training_config.optimization.batch_size)
    _LOGGER.info("Checkpoint directory: %s", training_config.checkpoints.checkpoint_directory)


def _handle_keyboard_interrupt(args: argparse.Namespace, training_config: RhythmRefinerTrainingConfig) -> None:
    latest_checkpoint_path = training_config.checkpoints.checkpoint_directory / "latest.pt"
    if not latest_checkpoint_path.exists():
        print("Rhythm refiner training interrupted. No latest checkpoint exists yet; resume command is unavailable.")
        return

    command = _resume_command(args, training_config=training_config, checkpoint_path=latest_checkpoint_path)
    print("Rhythm refiner training interrupted.")
    print(f"Resume command:\n{command}")


def _resume_command(
    args: argparse.Namespace,
    *,
    training_config: RhythmRefinerTrainingConfig,
    checkpoint_path: Path,
) -> str:
    command = [
        "uv",
        "run",
        "python",
        "scripts/train_refiner.py",
        "--data-dir",
        str(args.data_dir),
        "--ingestion-config",
        str(args.ingestion_config),
        "--segmentation-config",
        str(args.segmentation_config),
        "--tokenization-config",
        str(args.tokenization_config),
        "--training-config",
        str(args.training_config),
        "--checkpoint-dir",
        str(training_config.checkpoints.checkpoint_directory),
        "--resume-checkpoint",
        str(checkpoint_path),
        "--mlflow-db",
        str(args.mlflow_db),
        "--device",
        training_config.runtime.device,
        "--epochs",
        str(training_config.optimization.epochs),
        "--batch-size",
        str(training_config.optimization.batch_size),
        "--num-workers",
        str(training_config.runtime.num_workers),
        "--log-level",
        args.log_level,
    ]
    command.extend(_optional_value_argument("--learning-rate", args.learning_rate))
    command.extend(_optional_value_argument("--weight-decay", args.weight_decay))
    command.extend(_optional_value_argument("--validation-fraction", args.validation_fraction))
    command.extend(_optional_value_argument("--split-seed", args.split_seed))
    command.extend(_optional_value_argument("--window-bars", args.window_bars))
    command.extend(_optional_value_argument("--stride-bars", args.stride_bars))
    command.extend(_optional_path_argument("--difficulty-labels", args.difficulty_labels))
    command.extend(_optional_value_argument("--mlflow-experiment-name", args.mlflow_experiment_name))
    command.extend(_optional_value_argument("--mlflow-run-name", args.mlflow_run_name))
    if training_config.mlflow.enable_mlflow:
        command.extend(_optional_value_argument("--mlflow-run-id", _resume_mlflow_run_id(args, training_config)))
    if args.whole_file_segments:
        command.append("--whole-file-segments")
    if args.no_progress:
        command.append("--no-progress")
    if not training_config.mlflow.enable_mlflow:
        command.append("--disable-mlflow")
    return shlex.join(command)


def _optional_path_argument(name: str, value: Path | None) -> list[str]:
    if value is None:
        return []
    return [name, str(value)]


def _optional_value_argument(name: str, value: int | float | str | None) -> list[str]:
    if value is None:
        return []
    return [name, str(value)]


def _resume_mlflow_run_id(args: argparse.Namespace, training_config: RhythmRefinerTrainingConfig) -> str | None:
    if args.mlflow_run_id is not None:
        return str(args.mlflow_run_id)
    if training_config.mlflow.mlflow_run_id is not None:
        return training_config.mlflow.mlflow_run_id
    return read_mlflow_run_id(training_config.checkpoints.checkpoint_directory)


def _print_result(result: RhythmRefinerTrainingResult) -> None:
    if result.metrics:
        last_metric = result.metrics[-1]
        print(
            f"finished epochs={len(result.metrics)} "
            f"last_train_loss={last_metric.train_loss:.6f} "
            f"last_validation_loss={last_metric.validation_loss:.6f}"
        )
    print(f"latest_checkpoint={result.latest_checkpoint_path}")
    print(f"best_checkpoint={result.best_checkpoint_path}")


if __name__ == "__main__":
    main()
