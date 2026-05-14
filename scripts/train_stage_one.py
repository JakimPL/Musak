from __future__ import annotations

import argparse
from pathlib import Path

import torch

from musak_model.data.config import load_difficulty_labels, load_segmentation_config
from musak_model.paths import (
    CONDITIONING_CONFIG_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_MLFLOW_DIR,
    DEFAULT_STAGE_ONE_CHECKPOINT_DIR,
    INGESTION_CONFIG_PATH,
    SEGMENTATION_CONFIG_PATH,
    TOKENIZATION_CONFIG_PATH,
    TRAINING_CONFIG_PATH,
)
from musak_model.tokens.config import TokenizationConfig
from musak_model.training.config import TrainingConfig
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.trainer import TrainingResult, train_stage_one


def main() -> None:
    args = _parse_args()
    ingestion_config = _build_ingestion_config(args)
    segmentation_config = load_segmentation_config(
        args.segmentation_config,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
    )
    tokenization_config = TokenizationConfig.load(args.tokenization_config)
    training_config = _build_training_config(args)
    result = train_stage_one(
        args.data_dir,
        ingestion_config=ingestion_config,
        segmentation_config=segmentation_config,
        training_config=training_config,
        tokenization_config=tokenization_config,
        conditioning_config_path=args.conditioning_config,
    )
    _print_result(result)


class _TrainHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Musak Stage 1 autoregressive model. The data directory is the dataset root; "
            "when --processed-dir is set, training looks for reusable artifacts under "
            "<processed-dir>/<data-dir.name>/ before falling back to raw MusicXML."
        ),
        epilog=(
            "Examples:\n"
            "  uv run python scripts/train_stage_one.py --data-dir data/PDMX --processed-dir processed\n"
            "  uv run python scripts/train_stage_one.py --data-dir data/PDMX --processed-dir processed "
            "--tokenization-config musak_model/configs/tokens/tokenization.yml\n\n"
            "Artifact lookup:\n"
            "  Encoded artifacts are reused only from encoded/<tokenizer-hash>/ when tokenizer.json matches "
            "the active tokenization config.\n"
            "  If matching encoded artifacts are unavailable, parsed JSON artifacts are used when present.\n"
            "  If no usable processed artifacts exist, MusicXML files are parsed on the fly."
        ),
        formatter_class=_TrainHelpFormatter,
    )
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
    parser.add_argument("--training-config", type=Path, default=TRAINING_CONFIG_PATH, help="Training YAML config.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory for training checkpoints.")
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
    parser.add_argument("--use-conditioning", action="store_true", help="Enable conditioning inputs during training.")
    return parser.parse_args()


def _build_ingestion_config(args: argparse.Namespace) -> IngestionConfig:
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
