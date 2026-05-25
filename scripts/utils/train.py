from __future__ import annotations

import argparse
import logging
import shlex
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import torch

from musak_model.data.config import SegmentationMode, load_difficulty_labels, load_segmentation_config
from musak_model.paths import (
    CONDITIONING_CONFIG_PATH,
    DEFAULT_FINETUNING_CHECKPOINT_DIR,
    DEFAULT_MLFLOW_DIR,
    DEFAULT_PRETRAINING_CHECKPOINT_DIR,
    FINETUNING_CONFIG_PATH,
    INGESTION_CONFIG_PATH,
    PRETRAINING_CONFIG_PATH,
    SEGMENTATION_CONFIG_PATH,
    TOKENIZATION_CONFIG_PATH,
)
from musak_model.tokens.config import TokenizationConfig
from musak_model.training.config import (
    CheckpointConfig,
    FinetuningCheckpointConfig,
    FinetuningTrainingConfig,
    MlflowConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainingConditioningConfig,
    TrainingConfig,
)
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.stages.finetuning import finetune
from musak_model.training.stages.pretraining import TrainingResult, pretrain

_LOGGER = logging.getLogger(__name__)


class TrainingStage(StrEnum):
    PRETRAINING = "pretrain"
    FINETUNING = "finetune"


class TrainHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def run_training(stage: TrainingStage) -> None:
    args = parse_training_args(stage)
    configure_logging(args.log_level)
    validate_training_paths(args)
    source_directory = training_source_dir(args)
    ingestion_config = build_ingestion_config(args)
    segmentation_config = load_segmentation_config(
        args.segmentation_config,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
        mode=SegmentationMode.WHOLE_FILE if args.whole_file_segments else None,
    )
    tokenization_config = TokenizationConfig.load(args.tokenization_config)
    match stage:
        case TrainingStage.PRETRAINING:
            training_config = build_pretraining_training_config(args)
            if should_skip_pretraining(args=args, training_config=training_config):
                _LOGGER.info("Skipping pretraining because checkpoint files already exist.")
                print("Skipping pretraining because checkpoint files already exist.")
                return

            log_training_start(
                stage,
                args=args,
                ingestion_config=ingestion_config,
                training_config=training_config,
            )
            result = run_training_safely(
                lambda: pretrain(
                    source_directory,
                    ingestion_config=ingestion_config,
                    segmentation_config=segmentation_config,
                    training_config=training_config,
                    tokenization_config=tokenization_config,
                    conditioning_config_path=args.conditioning_config,
                    show_progress=not args.no_progress,
                    allow_raw_fallback=True,
                ),
                stage=stage,
                args=args,
                training_config=training_config,
            )
        case TrainingStage.FINETUNING:
            training_config = build_finetuning_training_config(args)
            validate_finetune_checkpoint(training_config)
            log_training_start(
                stage,
                args=args,
                ingestion_config=ingestion_config,
                training_config=training_config,
            )
            result = run_training_safely(
                lambda: finetune(
                    source_directory,
                    ingestion_config=ingestion_config,
                    segmentation_config=segmentation_config,
                    training_config=training_config,
                    tokenization_config=tokenization_config,
                    conditioning_config_path=args.conditioning_config,
                    show_progress=not args.no_progress,
                    allow_raw_fallback=True,
                ),
                stage=stage,
                args=args,
                training_config=training_config,
            )

    print_training_result(result)


def should_skip_pretraining(*, args: argparse.Namespace, training_config: TrainingConfig) -> bool:
    existing_checkpoints = existing_training_checkpoints(training_config)
    if args.resume_checkpoint is not None or args.overwrite or not existing_checkpoints:
        return False

    checkpoint_list = ", ".join(str(path) for path in existing_checkpoints)
    _LOGGER.warning("Pretraining checkpoint file(s) already exist: %s", checkpoint_list)
    prompt = (
        "Pretraining checkpoint file(s) already exist:\n"
        f"{checkpoint_list}\n"
        "Overwrite and run pretraining anyway? [y/N]: "
    )
    answer = input(prompt).strip().lower()
    return answer not in {"y", "yes"}


def existing_training_checkpoints(training_config: TrainingConfig) -> tuple[Path, ...]:
    checkpoint_dir = training_config.checkpoints.checkpoint_directory
    candidates = (checkpoint_dir / "latest.pt", checkpoint_dir / "best.pt")
    return tuple(path for path in candidates if path.exists())


def validate_finetune_checkpoint(training_config: FinetuningTrainingConfig) -> None:
    checkpoint_path = training_config.checkpoints.pretraining_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage-one pretrain checkpoint does not exist: {checkpoint_path}")


def run_training_safely(
    train: Callable[[], TrainingResult],
    *,
    stage: TrainingStage,
    args: argparse.Namespace,
    training_config: TrainingConfig,
) -> TrainingResult:
    try:
        return train()
    except KeyboardInterrupt as exception:
        handle_keyboard_interrupt(stage=stage, args=args, training_config=training_config)
        raise SystemExit(130) from exception


def handle_keyboard_interrupt(
    *,
    stage: TrainingStage,
    args: argparse.Namespace,
    training_config: TrainingConfig,
) -> None:
    latest_checkpoint_path = training_config.checkpoints.checkpoint_directory / "latest.pt"
    _LOGGER.warning("%s training interrupted by KeyboardInterrupt", stage.value)
    _LOGGER.warning("Checkpoint directory: %s", training_config.checkpoints.checkpoint_directory)
    if not latest_checkpoint_path.exists():
        _LOGGER.warning("No latest checkpoint exists yet; resume command is unavailable.")
        print("Training interrupted. No latest checkpoint exists yet; resume command is unavailable.")
        return

    command = resume_command(
        stage=stage, args=args, training_config=training_config, checkpoint_path=latest_checkpoint_path
    )
    _LOGGER.warning("Resume command: %s", command)
    print("Training interrupted.")
    print(f"Resume command:\n{command}")


def resume_command(
    *,
    stage: TrainingStage,
    args: argparse.Namespace,
    training_config: TrainingConfig,
    checkpoint_path: Path,
) -> str:
    command = ["uv", "run", "python", _executable(stage)]
    command.extend(_optional_path_argument("--data-dir", args.data_dir))
    command.extend(
        [
            "--ingestion-config",
            str(args.ingestion_config),
            "--segmentation-config",
            str(args.segmentation_config),
            "--tokenization-config",
            str(args.tokenization_config),
            "--conditioning-config",
            str(args.conditioning_config),
            "--training-config",
            str(args.training_config),
            "--checkpoint-dir",
            str(training_config.checkpoints.checkpoint_directory),
            "--resume-checkpoint",
            str(checkpoint_path),
            "--mlflow-dir",
            str(args.mlflow_dir),
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
    )
    command.extend(_optional_value_argument("--learning-rate", args.learning_rate))
    command.extend(_optional_value_argument("--weight-decay", args.weight_decay))
    command.extend(_optional_value_argument("--validation-fraction", args.validation_fraction))
    command.extend(_optional_value_argument("--split-seed", args.split_seed))
    command.extend(_optional_value_argument("--window-bars", args.window_bars))
    command.extend(_optional_value_argument("--stride-bars", args.stride_bars))
    if args.whole_file_segments:
        command.append("--whole-file-segments")
    if training_config.checkpoints.save_all_epochs:
        command.append("--save-all-epochs")
    command.extend(_optional_path_argument("--difficulty-labels", args.difficulty_labels))
    command.extend(_optional_value_argument("--mlflow-experiment-name", args.mlflow_experiment_name))
    command.extend(_optional_value_argument("--mlflow-run-name", args.mlflow_run_name))
    if args.no_progress:
        command.append("--no-progress")
    if args.overwrite:
        command.append("--overwrite")
    if not training_config.mlflow.enable_mlflow:
        command.append("--disable-mlflow")
    if isinstance(training_config, FinetuningTrainingConfig):
        command.extend(["--pretrain-checkpoint", str(training_config.checkpoints.pretraining_checkpoint)])

    return shlex.join(command)


def _optional_path_argument(name: str, value: Path | None) -> list[str]:
    if value is None:
        return []

    return [name, str(value)]


def _optional_value_argument(name: str, value: int | float | str | None) -> list[str]:
    if value is None:
        return []

    return [name, str(value)]


def parse_training_args(stage: TrainingStage) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=_description(stage),
        epilog=_epilog(stage),
        formatter_class=TrainHelpFormatter,
    )
    add_common_training_arguments(parser, stage=stage)
    return parser.parse_args()


def add_common_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    stage: TrainingStage,
) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root, including the dataset name, for example data/PDMX.",
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
    if stage == TrainingStage.FINETUNING:
        parser.add_argument(
            "--pretrain-checkpoint",
            type=Path,
            default=None,
            help="Stage-one checkpoint whose model weights initialize finetune fine-tuning.",
        )
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument(
        "--save-all-epochs",
        action="store_true",
        default=None,
        help="Save epoch_0000.pt style checkpoints after every epoch.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow pretraining to overwrite existing checkpoints.")
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
    parser.add_argument(
        "--whole-file-segments",
        action="store_true",
        help="Train from whole-file exercise segments instead of windowed segments.",
    )
    parser.add_argument("--difficulty-labels", type=Path, default=None, help="Optional YAML difficulty-label mapping.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Minimum logging level.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")


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
            }.items()
            if value is not None
        }
    )


def validate_training_paths(args: argparse.Namespace) -> None:
    if args.data_dir is None:
        raise ValueError("training requires --data-dir")


def training_source_dir(args: argparse.Namespace) -> Path:
    return Path(args.data_dir)


def build_pretraining_training_config(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig.load(args.training_config)
    return config.model_copy(
        update=common_training_section_updates(
            args,
            config=config,
            default_checkpoint_dir=DEFAULT_PRETRAINING_CHECKPOINT_DIR,
        )
    )


def build_finetuning_training_config(args: argparse.Namespace) -> FinetuningTrainingConfig:
    config = FinetuningTrainingConfig.load(args.training_config)
    updates = common_training_section_updates(
        args,
        config=config,
        default_checkpoint_dir=DEFAULT_FINETUNING_CHECKPOINT_DIR,
    )
    checkpoint_config = updates["checkpoints"]
    if not isinstance(checkpoint_config, CheckpointConfig):
        raise TypeError("checkpoint update must be a CheckpointConfig")

    updates["checkpoints"] = FinetuningCheckpointConfig(
        checkpoint_directory=checkpoint_config.checkpoint_directory,
        resume_checkpoint=checkpoint_config.resume_checkpoint,
        save_all_epochs=checkpoint_config.save_all_epochs,
        pretraining_checkpoint=(
            args.pretrain_checkpoint
            if args.pretrain_checkpoint is not None
            else config.checkpoints.pretraining_checkpoint
        ),
    )
    return config.model_copy(update=updates)


def common_training_section_updates(
    args: argparse.Namespace,
    *,
    config: TrainingConfig,
    default_checkpoint_dir: Path,
) -> dict[str, OptimizationConfig | RuntimeConfig | TrainingConditioningConfig | CheckpointConfig | MlflowConfig]:
    return {
        "optimization": OptimizationConfig(
            epochs=args.epochs if args.epochs is not None else config.optimization.epochs,
            batch_size=args.batch_size if args.batch_size is not None else config.optimization.batch_size,
            learning_rate=args.learning_rate if args.learning_rate is not None else config.optimization.learning_rate,
            weight_decay=args.weight_decay if args.weight_decay is not None else config.optimization.weight_decay,
        ),
        "runtime": RuntimeConfig(
            num_workers=args.num_workers if args.num_workers is not None else config.runtime.num_workers,
            device=resolve_device(args.device or config.runtime.device),
        ),
        "conditioning": config.conditioning,
        "checkpoints": CheckpointConfig(
            checkpoint_directory=args.checkpoint_dir
            or config.checkpoints.checkpoint_directory
            or default_checkpoint_dir,
            resume_checkpoint=(
                args.resume_checkpoint if args.resume_checkpoint is not None else config.checkpoints.resume_checkpoint
            ),
            save_all_epochs=(
                args.save_all_epochs if args.save_all_epochs is not None else config.checkpoints.save_all_epochs
            ),
        ),
        "mlflow": MlflowConfig(
            enable_mlflow=not args.disable_mlflow and config.mlflow.enable_mlflow,
            mlflow_experiment_name=args.mlflow_experiment_name or config.mlflow.mlflow_experiment_name,
            mlflow_run_name=args.mlflow_run_name if args.mlflow_run_name is not None else config.mlflow.mlflow_run_name,
            mlflow_tracking_uri=str(args.mlflow_dir),
        ),
    }


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def log_training_start(
    stage: TrainingStage,
    *,
    args: argparse.Namespace,
    ingestion_config: IngestionConfig,
    training_config: TrainingConfig,
) -> None:
    _LOGGER.info("Starting %s training", stage.value)
    _LOGGER.info("Raw input directory: %s", args.data_dir)
    _LOGGER.info("Internal processed root: %s", ingestion_config.processed_root)
    _LOGGER.info("Training config: %s", args.training_config)
    _LOGGER.info("Segmentation config: %s", args.segmentation_config)
    _LOGGER.info("Whole-file segments: %s", args.whole_file_segments)
    _LOGGER.info("Tokenization config: %s", args.tokenization_config)
    _LOGGER.info("Conditioning config: %s", args.conditioning_config)
    _LOGGER.info("Device: %s", training_config.runtime.device)
    _LOGGER.info("Epochs: %s", training_config.optimization.epochs)
    _LOGGER.info("Batch size: %s", training_config.optimization.batch_size)
    _LOGGER.info("Workers: %s", training_config.runtime.num_workers)
    _LOGGER.info("Progress bars: %s", not args.no_progress)
    _LOGGER.info("Checkpoint directory: %s", training_config.checkpoints.checkpoint_directory)
    _LOGGER.info("Latest checkpoint target: %s", training_config.checkpoints.checkpoint_directory / "latest.pt")
    _LOGGER.info("Best checkpoint target: %s", training_config.checkpoints.checkpoint_directory / "best.pt")
    if isinstance(training_config, FinetuningTrainingConfig):
        _LOGGER.info("Stage-one checkpoint: %s", training_config.checkpoints.pretraining_checkpoint)


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
        case TrainingStage.PRETRAINING:
            return (
                "Train the Musak Stage 1 autoregressive model. Pass --data-dir data/PDMX; training reads "
                "processed/PDMX when matching artifacts exist."
            )
        case TrainingStage.FINETUNING:
            return (
                "Fine-tune the Musak Stage 2 conditioned autoregressive model from a Stage 1 checkpoint. "
                "Use the same explicit dataset directory policy as Stage 1."
            )


def _epilog(stage: TrainingStage) -> str:
    executable = _executable(stage)
    finetuning_extra = ""
    if stage == TrainingStage.FINETUNING:
        finetuning_extra = " --pretrain-checkpoint checkpoints/pretraining/best.pt"
    return (
        "Examples:\n"
        f"  uv run python {executable} --data-dir data/PDMX{finetuning_extra}\n"
        f"  uv run python {executable} --data-dir data/PDMX "
        "--tokenization-config musak_model/configs/tokens/tokenization.yml"
        f"{finetuning_extra}\n\n"
        "Artifact lookup:\n"
        "  The dataset name comes from --data-dir, for example PDMX from data/PDMX.\n"
        "  Encoded artifacts are reused from processed/<dataset>/encoded/<tokenizer-hash>/ when "
        "tokenizer.json matches the active tokenization config.\n"
        "  If matching encoded artifacts are unavailable, parsed JSON artifacts are used when present.\n"
        "  If no usable processed artifacts exist, MusicXML files from --data-dir are parsed for training."
    )


def _executable(stage: TrainingStage) -> str:
    match stage:
        case TrainingStage.PRETRAINING:
            return "scripts/pretrain.py"
        case TrainingStage.FINETUNING:
            return "scripts/finetune.py"


def _training_config_path(stage: TrainingStage) -> Path:
    match stage:
        case TrainingStage.PRETRAINING:
            return PRETRAINING_CONFIG_PATH
        case TrainingStage.FINETUNING:
            return FINETUNING_CONFIG_PATH
