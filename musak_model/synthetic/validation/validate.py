from __future__ import annotations

import logging
import tempfile
from itertools import product
from pathlib import Path

import pandas as pd

from musak_model.evaluation.diagnostics import diagnose_segment
from musak_model.harmony.vocabulary import ChordVocabularyConfig
from musak_model.mlflow import MlflowRun, flatten_params
from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.profile.loading import FigureProfileArtifacts, load_figure_profile_artifacts
from musak_model.synthetic.fitting.form.fit import FormFittingConfig
from musak_model.synthetic.inputs import load_synthetic_inputs
from musak_model.synthetic.validation.config import SyntheticValidationConfig
from musak_model.synthetic.validation.generation import GeneratedSample, generate_scale_samples
from musak_model.synthetic.validation.metrics import validation_metrics
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import ScaleType

_LOGGER = logging.getLogger(__name__)
_FIGURE_DIRECTORY_NAME = "figure"


def validate(config: SyntheticValidationConfig, *, tracking_root: Path | None = None) -> dict[str, float]:
    if config.figure_root is None:
        raise ValueError("SyntheticValidationConfig.figure_root must be set before validation")

    inputs = load_synthetic_inputs(config.figure_root)
    form_fitting = FormFittingConfig.load()
    artifacts = _reference_artifacts(config.figure_root)
    if artifacts is None:
        _LOGGER.warning("No figure/rhythm reference profile under %s; fidelity metrics skipped", config.figure_root)

    samples_by_scale = {
        scale_type: generate_scale_samples(inputs, config, scale_type=scale_type, form_fitting=form_fitting)
        for scale_type in config.scales
    }
    metrics = validation_metrics(
        samples_by_scale,
        config=config,
        artifacts=artifacts,
        chord_vocabulary=ChordVocabularyConfig.load(),
        duration_vocabulary=inputs.duration_vocabulary,
        rhythm_config=NGramAnalysisConfig.load().rhythm_analysis,
    )

    with MlflowRun(config.mlflow, tracking_root=tracking_root) as run:
        run.log_params(flatten_params({"validation": config.model_dump(mode="json")}))
        run.log_metrics(metrics)
        run.log_dict(dict(metrics), "metrics.json")
        _log_artifacts(run, samples_by_scale, config=config, duration_vocabulary=inputs.duration_vocabulary)

    return metrics


def validate_sweep(config: SyntheticValidationConfig, *, tracking_root: Path | None = None) -> list[dict[str, float]]:
    if not config.sweep:
        return [validate(config, tracking_root=tracking_root)]

    keys = sorted(config.sweep)
    results: list[dict[str, float]] = []
    for combination in product(*(config.sweep[key] for key in keys)):
        update = dict(zip(keys, combination, strict=True))
        run_name = ",".join(f"{key}={value}" for key, value in update.items())
        _LOGGER.info("Validation sweep cell: %s", run_name)
        cell_config = config.model_copy(
            update={**update, "mlflow": config.mlflow.model_copy(update={"run_name": run_name})}
        )
        results.append(validate(cell_config, tracking_root=tracking_root))

    return results


def _reference_artifacts(figure_root: Path) -> FigureProfileArtifacts | None:
    encoded_directory = figure_root.parent if figure_root.name == _FIGURE_DIRECTORY_NAME else figure_root
    try:
        return load_figure_profile_artifacts(encoded_directory)
    except (FileNotFoundError, ValueError) as exception:
        _LOGGER.warning("Could not load reference profile under %s: %s", encoded_directory, exception)
        return None


def _log_artifacts(
    run: MlflowRun,
    samples_by_scale: dict[ScaleType, list[GeneratedSample]],
    *,
    config: SyntheticValidationConfig,
    duration_vocabulary: DurationVocabulary,
) -> None:
    if not run.enabled:
        return

    rows = _diagnostic_rows(samples_by_scale, duration_vocabulary=duration_vocabulary)
    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        diagnostics_path = staging / "sample_diagnostics.parquet"
        pd.DataFrame(rows).to_parquet(diagnostics_path)
        run.log_artifact(diagnostics_path, artifact_path="diagnostics")
        for path in _render_samples(
            samples_by_scale, config=config, duration_vocabulary=duration_vocabulary, staging=staging
        ):
            run.log_artifact(path, artifact_path="samples")


def _diagnostic_rows(
    samples_by_scale: dict[ScaleType, list[GeneratedSample]],
    *,
    duration_vocabulary: DurationVocabulary,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scale_type, samples in samples_by_scale.items():
        for sample in samples:
            row: dict[str, object] = {
                "scale_type": scale_type.value,
                "seed": sample.seed,
                "render_error": sample.render_error,
                "decode_error": None,
            }
            if sample.segment is not None:
                try:
                    row.update(diagnose_segment(sample.segment, duration_vocabulary=duration_vocabulary).model_dump())
                except ValueError as exception:
                    row["decode_error"] = str(exception)

            rows.append(row)

    return rows


def _render_samples(
    samples_by_scale: dict[ScaleType, list[GeneratedSample]],
    *,
    config: SyntheticValidationConfig,
    duration_vocabulary: DurationVocabulary,
    staging: Path,
) -> list[Path]:
    from musak_model.decoder import write_segment

    rendered = [sample for samples in samples_by_scale.values() for sample in samples if sample.segment is not None][
        : config.sample_render_count
    ]
    paths: list[Path] = []
    for sample in rendered:
        segment = sample.segment
        if segment is None:
            continue

        stem = f"{sample.scale_type.value}_seed{sample.seed}"
        for format_name, suffix in (("musicxml", ".musicxml"), ("midi", ".mid")):
            try:
                paths.append(
                    write_segment(
                        segment,
                        duration_vocabulary=duration_vocabulary,
                        path=staging / f"{stem}{suffix}",
                        format_name=format_name,
                    )
                )
            except (ValueError, OSError) as exception:
                _LOGGER.warning("Could not render %s as %s: %s", stem, format_name, exception)

    return paths
