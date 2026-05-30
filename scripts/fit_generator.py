import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.n_grams.profile.artifacts import figure_artifact_paths, figure_artifact_paths_from_root
from musak_model.paths import DEFAULT_PROCESSED_ROOT
from musak_model.synthetic.fitting.artifacts import FITTED_GENERATOR_CONFIG_NAME
from musak_model.synthetic.fitting.chord import ChordFitConfig
from musak_model.synthetic.fitting.fit import fit_generator_config
from musak_model.synthetic.processes.accent import AccentFieldConfig
from musak_model.synthetic.processes.pitch import RegisterCurveConfig
from scripts.extract_figures import resolve_encoded_directory

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
    try:
        figure_directory = _resolve_figure_directory(args)
        register_default = _load_register_config(args.register_config)
        accent_default = _load_accent_config(args.accent_config)
        chord_fit = _load_chord_fit_config(args.chord_fit_config)
        chord_vocabulary = ChordVocabularyConfig.load()
    except (FileNotFoundError, ValueError) as exception:
        _LOGGER.error("Generator fitting input is invalid: %s", exception)
        raise SystemExit(_EXIT_FAILURE) from exception

    output_path = figure_artifact_paths_from_root(figure_directory).all_directory / FITTED_GENERATOR_CONFIG_NAME
    _LOGGER.info("Starting generator fitting")
    _LOGGER.info("Figure directory: %s", figure_directory)
    _LOGGER.info("Grid denominator: %s", args.grid_denominator)
    _LOGGER.info("Register config: %s", args.register_config or "default")
    _LOGGER.info("Accent config: %s", args.accent_config or "default")
    _LOGGER.info("Fitted generator output: %s", output_path)
    try:
        fitted = fit_generator_config(
            figure_directory,
            register_default=register_default,
            accent_default=accent_default,
            chord_fit=chord_fit,
            chord_vocabulary=chord_vocabulary,
            grid_denominator=args.grid_denominator,
        )
    except (FileNotFoundError, ValueError) as exception:
        _LOGGER.error("Generator fitting failed: %s", exception)
        _LOGGER.error("Run figure analysis first to populate register and rhythm statistics.")
        raise SystemExit(_EXIT_FAILURE) from exception

    fitted.write(output_path)
    _LOGGER.info("Register overrides fitted: %s", len(fitted.register_overrides))
    _LOGGER.info("Accent overrides fitted: %s", len(fitted.accent_overrides))
    _LOGGER.info("Chord transition models fitted: %s", len(fitted.chord_transitions))
    _LOGGER.info("Figure-by-chord entries fitted: %s", len(fitted.chord_figure.entries))
    _LOGGER.info("Fitted generator config written to %s", output_path)


def _resolve_figure_directory(args: argparse.Namespace) -> Path:
    figure_directory: Path | None = args.figure_directory
    if figure_directory is not None:
        return figure_directory

    encoded_directory = resolve_encoded_directory(
        data_directory=args.data_dir,
        processed_root=args.processed_root,
        encoded_directory=args.encoded_directory,
    )
    return figure_artifact_paths(encoded_directory).root_directory


def _load_register_config(path: Path | None) -> RegisterCurveConfig:
    return RegisterCurveConfig.load() if path is None else RegisterCurveConfig.load(path)


def _load_accent_config(path: Path | None) -> AccentFieldConfig:
    return AccentFieldConfig.load() if path is None else AccentFieldConfig.load(path)


def _load_chord_fit_config(path: Path | None) -> ChordFitConfig:
    return ChordFitConfig.load() if path is None else ChordFitConfig.load(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit register and accent generator overrides from persisted corpus statistics and write the "
        "fitted-generator artifact next to the figure profile.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Dataset root. Resolves <processed-root>/<data-dir.name>/encoded/<run>/figure as the figure directory.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Root directory for processed dataset artifacts.",
    )
    parser.add_argument(
        "--encoded-directory",
        type=Path,
        help="Specific encoded run directory; required only when the processed dataset has multiple encoded runs.",
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        help="Figure artifact root holding all/, register/ and rhythm/. Overrides --data-dir resolution when set.",
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
        "--chord-fit-config",
        type=Path,
        help="Chord-fit YAML config (prior_count, functional_strength, self_transition_bias) for the empirical "
        "transition smoothing prior.",
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
