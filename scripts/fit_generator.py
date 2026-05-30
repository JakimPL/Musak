import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from musak_model.synthetic.fitting.artifacts import FITTED_GENERATOR_CONFIG_NAME
from musak_model.synthetic.fitting.fit import fit_generator_config
from musak_model.synthetic.processes.accent import AccentFieldConfig
from musak_model.synthetic.processes.pitch import RegisterCurveConfig

_LOGGER = logging.getLogger(__name__)
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}
_EXIT_FAILURE: Final[int] = 1


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)
    output_path = args.figure_directory / FITTED_GENERATOR_CONFIG_NAME
    _LOGGER.info("Starting generator fitting")
    _LOGGER.info("Figure directory: %s", args.figure_directory)
    _LOGGER.info("Grid denominator: %s", args.grid_denominator)
    _LOGGER.info("Register config: %s", args.register_config or "default")
    _LOGGER.info("Accent config: %s", args.accent_config or "default")
    _LOGGER.info("Fitted generator output: %s", output_path)
    try:
        fitted = fit_generator_config(
            args.figure_directory,
            register_default=_load_register_config(args.register_config),
            accent_default=_load_accent_config(args.accent_config),
            grid_denominator=args.grid_denominator,
        )
    except (FileNotFoundError, ValueError) as exception:
        _LOGGER.error("Generator fitting failed: %s", exception)
        _LOGGER.error("Run scripts/extract_figures.py first to populate register and rhythm statistics.")
        raise SystemExit(_EXIT_FAILURE) from exception

    fitted.write(output_path)
    _LOGGER.info("Register overrides fitted: %s", len(fitted.register_overrides))
    _LOGGER.info("Accent overrides fitted: %s", len(fitted.accent_overrides))
    _LOGGER.info("Fitted generator config written to %s", output_path)


def _load_register_config(path: Path | None) -> RegisterCurveConfig:
    return RegisterCurveConfig.load() if path is None else RegisterCurveConfig.load(path)


def _load_accent_config(path: Path | None) -> AccentFieldConfig:
    return AccentFieldConfig.load() if path is None else AccentFieldConfig.load(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit register and accent generator overrides from persisted corpus statistics and write the "
        "fitted-generator artifact next to the figure profile.",
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        required=True,
        help="Figure artifact root holding all/, register/ and rhythm/ statistics (e.g. <encoded>/figure).",
    )
    parser.add_argument(
        "--grid-denominator",
        type=int,
        required=True,
        help="Onset-position grid denominator to fit the accent field at; align with the generation grid.",
    )
    parser.add_argument(
        "--register-config",
        type=Path,
        help="Register-curve YAML config providing fit defaults (arch_basis_count, arch_decay).",
    )
    parser.add_argument(
        "--accent-config",
        type=Path,
        help="Accent-field YAML config providing fit defaults (envelope parameters).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Minimum logging level.",
    )
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=_LOG_LEVELS[level],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    main()
