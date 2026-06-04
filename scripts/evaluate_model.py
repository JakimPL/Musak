from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import torch

from musak_model.evaluation import GenerationSuiteEvaluator
from musak_model.mlflow import MlflowRun, MlflowRunConfig, flatten_params, sqlite_tracking_uri
from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import ModelConfig
from musak_model.paths import (
    CONDITIONING_CONFIG_PATH,
    DEFAULT_FINETUNING_CHECKPOINT_DIRECTORY,
    DEFAULT_MLFLOW_DB_PATH,
    DEFAULT_PRETRAINING_CHECKPOINT_DIRECTORY,
    FINETUNING_CONFIG_PATH,
    GENERATION_EVALUATION_CONFIG_PATH,
    INGESTION_CONFIG_PATH,
    PRETRAINING_CONFIG_PATH,
    TOKENIZATION_CONFIG_PATH,
)
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.checkpoint import load_model_weights
from musak_model.training.config import FinetuningTrainingConfig, GenerationEvaluationConfig, TrainingConfig
from musak_model.training.ingestion.config import IngestionConfig
from musak_model.training.stages.figure_profiles import load_generation_figure_profile_artifacts
from scripts.utils.logger import DEFAULT_LOG_LEVEL, LOG_LEVEL_CHOICES, configure_logging
from scripts.utils.train import TrainingStage, resolve_device

_LOGGER = logging.getLogger(__name__)

_CHECKPOINT_FILE_NAME: Final[str] = "best.pt"
_GENERATION_ARTIFACT_PATH: Final[str] = "generation/evaluation"


@dataclass(frozen=True)
class EvaluationStageDefaults:
    checkpoint_path: Path
    training_config_path: Path
    mlflow_experiment_name: str


def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)
    evaluate_model(args)


def evaluate_model(args: argparse.Namespace) -> None:
    stage = evaluation_stage(args.stage)
    defaults = evaluation_stage_defaults(stage)
    checkpoint_path = args.checkpoint or defaults.checkpoint_path
    training_config_path = args.training_config or defaults.training_config_path
    mlflow_experiment_name = args.mlflow_experiment_name or defaults.mlflow_experiment_name
    device = torch.device(resolve_device(args.device))
    generation_config = _generation_config(
        args.generation_evaluation_config, seed=args.seed, temperature=args.temperature
    )
    training_config = _training_config(
        stage=stage,
        training_config_path=training_config_path,
        generation_config=generation_config,
        device=args.device,
    )
    tokenization_config = TokenizationConfig.load(args.tokenization_config)
    token_vocabulary = _token_vocabulary(tokenization_config)
    model_config = _model_config(
        token_vocabulary=token_vocabulary,
        training_config=training_config,
        conditioning_config_path=args.conditioning_config,
    )
    model = _load_model(checkpoint_path, stage=stage, model_config=model_config, device=device)
    figure_profile_artifacts = load_generation_figure_profile_artifacts(
        source_directory=args.data_dir,
        ingestion_config=IngestionConfig.load(args.ingestion_config),
        tokenization_config=tokenization_config,
    )
    evaluator = GenerationSuiteEvaluator(
        config=training_config.generation_evaluation,
        conditioning=training_config.conditioning,
        model_config=model_config,
        token_vocabulary=token_vocabulary,
        duration_vocabulary=token_vocabulary.duration_vocabulary,
        include_bar_count_control=False,
        figure_profile_artifacts=figure_profile_artifacts,
        show_progress=not args.no_progress,
    )
    run_name = args.mlflow_run_name or default_run_name(
        stage=stage,
        data_dir=args.data_dir,
        checkpoint=checkpoint_path,
        config=training_config.generation_evaluation,
    )
    tracking_uri = None if args.disable_mlflow else sqlite_tracking_uri(args.mlflow_db)
    with MlflowRun(
        MlflowRunConfig(
            enabled=not args.disable_mlflow,
            experiment_name=mlflow_experiment_name,
            run_name=run_name,
            run_id=args.mlflow_run_id,
            tracking_uri=args.mlflow_tracking_uri or tracking_uri,
        )
    ) as run:
        run.log_params(
            flatten_params(
                {
                    "evaluation": _evaluation_param_values(
                        args,
                        stage=stage,
                        checkpoint=checkpoint_path,
                        training_config_path=training_config_path,
                        generation_evaluation_config_path=args.generation_evaluation_config,
                        run_name=run_name,
                    ),
                    "generation": training_config.generation_evaluation.model_dump(mode="json"),
                    "model": model_config.model_dump(mode="json"),
                }
            )
        )
        _LOGGER.info("Running %s generation evaluation", stage.value)
        result = evaluator.evaluate_result(model, device=device)
        run.log_metrics(result.metrics, step=0)
        with TemporaryDirectory(prefix=f"musak-evaluate-{stage.value}-") as temporary_directory:
            artifact_directory = Path(temporary_directory)
            evaluator.write_artifacts(result, output_directory=artifact_directory)
            run.log_artifacts(artifact_directory, artifact_path=_GENERATION_ARTIFACT_PATH)

    print(f"stage={stage.value}")
    print(f"generation_metrics={len(result.metrics)}")
    print(f"soft_samples={training_config.generation_evaluation.soft_sample_count}")
    print(f"hard_samples={training_config.generation_evaluation.hard_sample_count}")
    print(f"bar_count={training_config.generation_evaluation.bar_count}")


def evaluation_stage(raw_stage: str) -> TrainingStage:
    return TrainingStage(raw_stage)


def evaluation_stage_defaults(stage: TrainingStage) -> EvaluationStageDefaults:
    match stage:
        case TrainingStage.PRETRAINING:
            return EvaluationStageDefaults(
                checkpoint_path=DEFAULT_PRETRAINING_CHECKPOINT_DIRECTORY / _CHECKPOINT_FILE_NAME,
                training_config_path=PRETRAINING_CONFIG_PATH,
                mlflow_experiment_name="musak-evaluate-pretrain",
            )
        case TrainingStage.FINETUNING:
            return EvaluationStageDefaults(
                checkpoint_path=DEFAULT_FINETUNING_CHECKPOINT_DIRECTORY / _CHECKPOINT_FILE_NAME,
                training_config_path=FINETUNING_CONFIG_PATH,
                mlflow_experiment_name="musak-evaluate-finetune",
            )


def default_run_name(
    *,
    stage: TrainingStage,
    data_dir: Path,
    checkpoint: Path,
    config: GenerationEvaluationConfig,
) -> str:
    checkpoint_name = checkpoint.stem
    return (
        f"eval-{stage.value}-{checkpoint_name}-"
        f"{data_dir.name}-"
        f"gen{config.bar_count}b-"
        f"{config.soft_sample_count}s{config.hard_sample_count}h-"
        f"seed{config.seed}"
    )


def generation_config_with_overrides(
    config: GenerationEvaluationConfig,
    *,
    seed: int | None,
    temperature: float | None,
) -> GenerationEvaluationConfig:
    updates = {
        key: value
        for key, value in {
            "enabled": True,
            "seed": seed,
            "temperature": temperature,
        }.items()
        if value is not None
    }
    return GenerationEvaluationConfig.model_validate({**config.model_dump(), **updates})


def _training_config(
    *,
    stage: TrainingStage,
    training_config_path: Path,
    generation_config: GenerationEvaluationConfig,
    device: str,
) -> TrainingConfig:
    config = _load_training_config(stage=stage, training_config_path=training_config_path)
    return config.model_copy(
        update={
            "runtime": config.runtime.model_copy(update={"device": resolve_device(device)}),
            "generation_evaluation": generation_config,
        }
    )


def _generation_config(
    generation_evaluation_config_path: Path,
    *,
    seed: int | None,
    temperature: float | None,
) -> GenerationEvaluationConfig:
    return generation_config_with_overrides(
        GenerationEvaluationConfig.load(generation_evaluation_config_path),
        seed=seed,
        temperature=temperature,
    )


def _load_training_config(*, stage: TrainingStage, training_config_path: Path) -> TrainingConfig:
    match stage:
        case TrainingStage.PRETRAINING:
            return TrainingConfig.load(training_config_path)
        case TrainingStage.FINETUNING:
            return FinetuningTrainingConfig.load(training_config_path)


def _token_vocabulary(tokenization_config: TokenizationConfig) -> TokenVocabulary:
    return TokenVocabulary(DurationVocabulary(tokenization_config))


def _model_config(
    *,
    token_vocabulary: TokenVocabulary,
    training_config: TrainingConfig,
    conditioning_config_path: Path,
) -> ModelConfig:
    return ModelConfig.load(
        vocabulary_size=token_vocabulary.vocabulary_size,
        duration_vocabulary_size=token_vocabulary.duration_vocabulary.vocabulary_size(),
        output_mode=training_config.event_objective.mode,
        musical_auxiliary_targets=training_config.musical_auxiliary_targets,
        conditioning_config_path=conditioning_config_path,
    )


def _load_model(
    checkpoint_path: Path,
    *,
    stage: TrainingStage,
    model_config: ModelConfig,
    device: torch.device,
) -> HierarchicalAutoregressiveModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{stage.value} checkpoint does not exist: {checkpoint_path}")

    _LOGGER.info("Loading %s checkpoint: %s", stage.value, checkpoint_path)
    model = HierarchicalAutoregressiveModel(model_config)
    load_model_weights(checkpoint_path, model=model, device=device)
    model.to(device)
    return model


def _evaluation_param_values(
    args: argparse.Namespace,
    *,
    stage: TrainingStage,
    checkpoint: Path,
    training_config_path: Path,
    generation_evaluation_config_path: Path,
    run_name: str,
) -> dict[str, object]:
    return {
        "stage": stage.value,
        "data_dir": str(args.data_dir),
        "checkpoint": str(checkpoint),
        "training_config": str(training_config_path),
        "generation_evaluation_config": str(generation_evaluation_config_path),
        "tokenization_config": str(args.tokenization_config),
        "conditioning_config": str(args.conditioning_config),
        "ingestion_config": str(args.ingestion_config),
        "device": resolve_device(args.device),
        "run_name": run_name,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generation evaluation for a trained model checkpoint.")
    parser.add_argument(
        "stage",
        choices=tuple(stage.value for stage in TrainingStage),
        help="Training stage checkpoint type to evaluate.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root or dataset name used to resolve processed figure artifacts.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to evaluate.")
    parser.add_argument("--training-config", type=Path, default=None, help="Training YAML config.")
    parser.add_argument(
        "--generation-evaluation-config",
        type=Path,
        default=GENERATION_EVALUATION_CONFIG_PATH,
        help="Generation evaluation YAML config.",
    )
    parser.add_argument(
        "--tokenization-config",
        type=Path,
        default=TOKENIZATION_CONFIG_PATH,
        help="Tokenization YAML config.",
    )
    parser.add_argument(
        "--conditioning-config",
        type=Path,
        default=CONDITIONING_CONFIG_PATH,
        help="Conditioning YAML config.",
    )
    parser.add_argument("--ingestion-config", type=Path, default=INGESTION_CONFIG_PATH, help="Ingestion YAML config.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto", help="Evaluation device.")
    parser.add_argument("--seed", type=int, default=None, help="Override generation seed.")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--mlflow-db", type=Path, default=DEFAULT_MLFLOW_DB_PATH, help="Local MLflow SQLite database.")
    parser.add_argument("--mlflow-experiment-name", default=None, help="MLflow experiment name.")
    parser.add_argument("--mlflow-run-name", default=None, help="Optional MLflow run name.")
    parser.add_argument("--mlflow-run-id", default=None, help="Attach evaluation logs to an existing MLflow run.")
    parser.add_argument("--mlflow-tracking-uri", default=None, help="Optional MLflow tracking URI override.")
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow logging.")
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVEL_CHOICES,
        default=DEFAULT_LOG_LEVEL,
        help="Minimum logging level.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
