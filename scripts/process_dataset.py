import argparse
from pathlib import Path

from musak_model.common.files import load_yaml_config
from musak_model.data.config import SegmentationConfig
from musak_model.paths import DEFAULT_PROCESSED_ROOT, SEGMENTATION_CONFIG_PATH, TOKENIZATION_CONFIG_PATH
from musak_model.processing.dataset import process_dataset
from musak_model.tokens.config import TokenizationConfig


def main() -> None:
    args = _parse_args()
    result = process_dataset(
        args.data_dir,
        processed_root=args.processed_dir,
        dataset_name=args.dataset_name or args.data_dir.name,
        segmentation=_build_segmentation_config(args),
        tokenization_config=TokenizationConfig.load(args.tokenization_config),
        stage=args.stage,
        difficulty_labels=_load_difficulty_labels(args.difficulty_labels),
        overwrite=args.overwrite,
    )
    print(f"parsed_manifest={result.parsed_manifest_path}")
    if result.encoded_manifest_path is not None:
        print(f"encoded_manifest={result.encoded_manifest_path}")
    if result.tokenizer_snapshot_path is not None:
        print(f"tokenizer_snapshot={result.tokenizer_snapshot_path}")
    print(f"parsed={result.parsed_count} " f"encoded={result.encoded_count} " f"errors={result.error_count}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a MusicXML dataset into reusable intermediate artifacts.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--stage", choices=("parsed", "encoded", "all"), default="all")
    parser.add_argument("--segmentation-config", type=Path, default=SEGMENTATION_CONFIG_PATH)
    parser.add_argument("--tokenization-config", type=Path, default=TOKENIZATION_CONFIG_PATH)
    parser.add_argument("--window-bars", type=int, default=None)
    parser.add_argument("--stride-bars", type=int, default=None)
    parser.add_argument("--difficulty-labels", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


if __name__ == "__main__":
    main()
