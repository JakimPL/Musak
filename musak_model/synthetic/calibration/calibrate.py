from musak_model.synthetic.calibration.assembly import build_calibration_generator, load_reference_counts
from musak_model.synthetic.calibration.config import CalibrationConfig
from musak_model.synthetic.calibration.results import write_sweep_results
from musak_model.synthetic.calibration.schema import SweepResult
from musak_model.synthetic.calibration.selection import select_tilts, write_selected_substitution_config
from musak_model.synthetic.calibration.sweep import run_sweep


def calibrate(config: CalibrationConfig) -> list[SweepResult]:
    results = run_sweep(
        generator=build_calibration_generator(config),
        reference_counts=load_reference_counts(config),
        config=config,
    )
    write_sweep_results(results, config.output_path)
    selection = select_tilts(
        results,
        lambda_curve=config.lambda_curve,
        lambda_harmonic=config.lambda_harmonic,
        lambda_accent=config.lambda_accent,
        threshold=config.target_total_variation_distance,
    )
    write_selected_substitution_config(
        selection,
        config,
        config.output_path.with_name(f"{config.output_path.stem}_selected.json"),
    )
    return results
