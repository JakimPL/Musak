import logging

from musak_model.synthetic.calibration import CalibrationConfig, calibrate
from scripts.utils.logger import configure_logging

_LOGGER = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    config = CalibrationConfig.load()
    _LOGGER.info("Calibrating substitution tilts against %s", config.figure_root)
    results = calibrate(config)
    _LOGGER.info("Wrote %s calibration row(s) to %s", len(results), config.output_path)


if __name__ == "__main__":
    main()
