from __future__ import annotations

import argparse
from pathlib import Path

import torch

from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import ModelConfig
from musak_model.paths import CONDITIONING_CONFIG_PATH, MODEL_CONFIG_DIR, TOKENIZATION_CONFIG_PATH
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.checkpoint_migration import (
    CheckpointMigrationReport,
    migrate_checkpoint_to_model,
)


def main() -> None:
    args = _parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output checkpoint already exists: {args.output}; use --overwrite to overwrite it")

    torch.manual_seed(args.seed)
    tokenization_config = TokenizationConfig.load(args.tokenization_config)
    token_vocabulary = TokenVocabulary(DurationVocabulary(tokenization_config))
    model_config = ModelConfig.load(
        vocabulary_size=token_vocabulary.vocabulary_size,
        config_directory=args.model_config_dir,
        conditioning_config_path=args.conditioning_config,
    )
    model = HierarchicalAutoregressiveModel(model_config)
    report = migrate_checkpoint_to_model(
        args.input,
        args.output,
        model=model,
        device=torch.device(args.device),
        preserve_optimizer_state=args.preserve_optimizer_state,
        allow_truncation=args.allow_truncation,
    )
    _print_report(report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt an older Musak model checkpoint to the current model shape by preserving matching weights "
            "and initializing missing or expanded tensors from a freshly initialized target model."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Source checkpoint path.")
    parser.add_argument("--output", type=Path, required=True, help="Adapted checkpoint output path.")
    parser.add_argument("--device", default="cpu", help="Device used for checkpoint loading.")
    parser.add_argument("--seed", type=int, default=0, help="Initialization seed for newly created weights.")
    parser.add_argument(
        "--tokenization-config",
        type=Path,
        default=TOKENIZATION_CONFIG_PATH,
        help="Tokenization config for the target model.",
    )
    parser.add_argument(
        "--conditioning-config",
        type=Path,
        default=CONDITIONING_CONFIG_PATH,
        help="Conditioning config for the target model.",
    )
    parser.add_argument(
        "--model-config-dir",
        type=Path,
        default=MODEL_CONFIG_DIR,
        help="Directory containing cnn.yml, gru.yml, and transformer.yml for the target model.",
    )
    parser.add_argument(
        "--preserve-optimizer-state",
        action="store_true",
        help="Keep optimizer state. By default it is reset because parameter shapes may have changed.",
    )
    parser.add_argument(
        "--allow-truncation",
        action="store_true",
        help="Allow source tensors larger than target tensors to be cropped to the overlapping slice.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output checkpoint if it exists.")
    return parser.parse_args()


def _print_report(report: CheckpointMigrationReport) -> None:
    print("checkpoint adapted")
    print(f"changed_tensors={len(report.changed_tensors)}")
    for tensor in report.changed_tensors:
        print(
            f"{tensor.action}: {tensor.key} "
            f"{tensor.source_shape if tensor.source_shape is not None else '-'} -> {tensor.target_shape}"
        )
    if report.ignored_source_keys:
        print(f"ignored_source_keys={len(report.ignored_source_keys)}")
        for key in report.ignored_source_keys:
            print(f"ignored: {key}")
    print(f"optimizer_state_preserved={report.optimizer_state_preserved}")


if __name__ == "__main__":
    main()
