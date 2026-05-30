from pathlib import Path

from musak_model.n_grams.profile.register.io import read_register_metadata, read_register_statistics
from musak_model.n_grams.profile.register.schema import register_artifact_paths_for_figure_root
from musak_model.n_grams.profile.rhythm.io import read_rhythm_counts
from musak_model.n_grams.profile.rhythm.schema import rhythm_artifact_paths_for_figure_root
from musak_model.synthetic.fitting.accent import fit_accent_overrides_from_rhythm_counts
from musak_model.synthetic.fitting.artifacts import FittedGeneratorConfig
from musak_model.synthetic.fitting.register import fit_register_overrides_from_statistics
from musak_model.synthetic.processes.accent import AccentFieldConfig
from musak_model.synthetic.processes.pitch import RegisterCurveConfig


def fit_generator_config(
    figure_root_directory: Path,
    *,
    register_default: RegisterCurveConfig,
    accent_default: AccentFieldConfig,
    grid_denominator: int,
) -> FittedGeneratorConfig:
    register_paths = register_artifact_paths_for_figure_root(figure_root_directory)
    metadata = read_register_metadata(register_paths.metadata_path)
    if metadata.arch_basis_count != register_default.arch_basis_count:
        raise ValueError(
            f"register statistics were computed with arch_basis_count={metadata.arch_basis_count}, "
            f"but the register config uses arch_basis_count={register_default.arch_basis_count}"
        )

    register_statistics = read_register_statistics(register_paths.statistics_path)
    rhythm_paths = rhythm_artifact_paths_for_figure_root(figure_root_directory)
    rhythm_counts = read_rhythm_counts(rhythm_paths.counts_path)
    return FittedGeneratorConfig(
        register_overrides=fit_register_overrides_from_statistics(register_statistics, default=register_default),
        accent_overrides=fit_accent_overrides_from_rhythm_counts(
            rhythm_counts, default=accent_default, grid_denominator=grid_denominator
        ),
    )
