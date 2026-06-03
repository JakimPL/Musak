import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from musak_model.n_grams.profile.artifacts import figure_artifact_paths
from musak_model.paths import DEFAULT_PROCESSED_ROOT
from musak_model.synthetic.validation.config import SyntheticValidationConfig
from musak_model.synthetic.validation.validate import validate_sweep
from scripts.extract_figures import resolve_encoded_directory

_LOGGER = logging.getLogger(__name__)
_LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
_EXIT_FAILURE: Final[int] = 1


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)
    try:
        figure_root = _resolve_figure_directory(args)
        config = _build_config(args, figure_root=figure_root)
    except (FileNotFoundError, ValueError) as exception:
        _LOGGER.error("Synthetic validation input is invalid: %s", exception)
        raise SystemExit(_EXIT_FAILURE) from exception

    _LOGGER.info("Validating synthetic generator on %s", figure_root)
    _LOGGER.info(
        "Scales: %s | samples per scale: %s", [scale.value for scale in config.scales], config.samples_per_scale
    )
    results = validate_sweep(config)
    _LOGGER.info("Completed %s validation run(s)", len(results))
    for metrics in results:
        for name in sorted(metrics):
            if name.endswith("/identity_total_variation_distance") or name.endswith("/strong_beat_chord_tone_fraction"):
                _LOGGER.info("  %s = %.4f", name, metrics[name])


def _resolve_figure_directory(args: argparse.Namespace) -> Path:
    if args.figure_directory is not None:
        return Path(args.figure_directory)

    encoded_directory = resolve_encoded_directory(
        data_directory=args.data_dir,
        processed_root=args.processed_root,
        encoded_directory=args.encoded_directory,
    )
    return Path(figure_artifact_paths(encoded_directory).root_directory)


def _build_config(args: argparse.Namespace, *, figure_root: Path) -> SyntheticValidationConfig:
    config = SyntheticValidationConfig.load() if args.config is None else SyntheticValidationConfig.load(args.config)
    mlflow_updates: dict[str, object] = {}
    if args.mlflow_experiment_name is not None:
        mlflow_updates["experiment_name"] = args.mlflow_experiment_name
    if args.mlflow_run_name is not None:
        mlflow_updates["run_name"] = args.mlflow_run_name
    if args.mlflow_tracking_uri is not None:
        mlflow_updates["tracking_uri"] = args.mlflow_tracking_uri
    if args.disable_mlflow:
        mlflow_updates["enabled"] = False

    updates: dict[str, object] = {"figure_root": figure_root, "mlflow": config.mlflow.model_copy(update=mlflow_updates)}
    if args.samples_per_scale is not None:
        updates["samples_per_scale"] = args.samples_per_scale

    return config.model_copy(update=updates)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render synthetic exercises at scale and log generation-quality metrics to MLflow."
    )
    parser.add_argument("--data-dir", type=Path, help="Dataset root; resolves the figure artifact directory.")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--encoded-directory", type=Path, help="Specific encoded run directory.")
    parser.add_argument("--figure-directory", type=Path, help="Figure artifact root (overrides --data-dir resolution).")
    parser.add_argument("--config", type=Path, help="Validation YAML config (defaults to the bundled validation.yml).")
    parser.add_argument("--samples-per-scale", type=int, help="Override the number of rendered samples per scale.")
    parser.add_argument("--mlflow-experiment-name", type=str)
    parser.add_argument("--mlflow-run-name", type=str)
    parser.add_argument("--mlflow-tracking-uri", type=str)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--log-level", choices=tuple(_LOG_LEVELS), default="INFO")
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=_LOG_LEVELS[level], format="%(asctime)s %(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    main()
