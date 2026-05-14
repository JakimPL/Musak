from __future__ import annotations

import argparse
from pathlib import Path

import torch

from musak_model.common.files import load_yaml_config
from musak_model.data.config import SegmentationConfig
from musak_model.paths import (
    CONDITIONING_CONFIG_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_MLFLOW_DIR,
    DEFAULT_STAGE_ONE_CHECKPOINT_DIR,
    INGESTION_CONFIG_PATH,
    SEGMENTATION_CONFIG_PATH,
    TRAINING_CONFIG_PATH,
)
from musak_model.training.config import TrainingConfig
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.trainer import TrainingResult, train_stage_one


def main() -> None:
    args = _parse_args()
    ingestion_config = _build_ingestion_config(args)
    segmentation_config = _build_segmentation_config(args)
    training_config = _build_training_config(args)
    result = train_stage_one(
        args.data_dir,
        ingestion_config=ingestion_config,
        segmentation_config=segmentation_config,
        training_config=training_config,
        conditioning_config_path=args.conditioning_config,
    )
    _print_result(result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Musak Stage 1 autoregressive model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ingestion-config", type=Path, default=INGESTION_CONFIG_PATH)
    parser.add_argument("--segmentation-config", type=Path, default=SEGMENTATION_CONFIG_PATH)
    parser.add_argument("--conditioning-config", type=Path, default=CONDITIONING_CONFIG_PATH)
    parser.add_argument("--training-config", type=Path, default=TRAINING_CONFIG_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--mlflow-dir", type=Path, default=DEFAULT_MLFLOW_DIR)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment-name", type=str, default=None)
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--window-bars", type=int, default=None)
    parser.add_argument("--stride-bars", type=int, default=None)
    parser.add_argument("--difficulty-labels", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--use-conditioning", action="store_true")
    return parser.parse_args()


def _build_ingestion_config(args: argparse.Namespace) -> IngestionConfig:
    config = IngestionConfig.load(args.ingestion_config)
    difficulty_labels = _load_difficulty_labels(args.difficulty_labels)
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


def _build_segmentation_config(args: argparse.Namespace) -> SegmentationConfig:
    config = SegmentationConfig.load(args.segmentation_config)
    return config.model_copy(
        update={
            key: value
            for key, value in {
                "window_bars": args.window_bars,
                "stride_bars": args.stride_bars,
            }.items()
            if value is not None
        }
    )


def _build_training_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig.load(args.training_config)
    updates = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "checkpoint_dir": args.checkpoint_dir or config.checkpoint_dir or DEFAULT_STAGE_ONE_CHECKPOINT_DIR,
        "resume_checkpoint": args.resume_checkpoint if args.resume_checkpoint is not None else config.resume_checkpoint,
        "device": _resolve_device(args.device or config.device),
        "use_conditioning": args.use_conditioning or config.use_conditioning,
        "enable_mlflow": not args.disable_mlflow and config.enable_mlflow,
        "mlflow_experiment_name": args.mlflow_experiment_name or config.mlflow_experiment_name,
        "mlflow_run_name": args.mlflow_run_name if args.mlflow_run_name is not None else config.mlflow_run_name,
        "mlflow_tracking_uri": str(args.mlflow_dir),
    }
    return config.model_copy(update={key: value for key, value in updates.items() if value is not None})


def _load_difficulty_labels(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None

    parsed = load_yaml_config(path)
    labels: dict[str, int] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise ValueError("difficulty labels must be a mapping of file stem to integer difficulty level")
        labels[key] = value

    return labels


def _resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _print_result(result: TrainingResult) -> None:
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


if __name__ == "__main__":
    main()
