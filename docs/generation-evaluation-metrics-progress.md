# Generation Evaluation Metrics Progress

This tracker is the implementation guardrail for `docs/generation-evaluation-metrics-plan.md`. Before changing code in
any phase, re-read `docs/guidelines.md` and keep the phase scoped to the roadmap below.

## Roadmap

| Phase | Status | Review Gate |
| --- | --- | --- |
| 1. Docs and baseline inventory | complete | Plan and tracker saved in docs. |
| 2. Refactor generation evaluation package | complete | Existing behavior preserved; generation package split by concern. |
| 3. Extend shared n-gram analysis config | complete | `analysis/n_grams.yml` and `NGramAnalysisConfig` own reference comparison parameters. |
| 4. Add shared reference-distribution figure metrics | accepted | Common/rare/novel, property, contour, and duration-shape metrics implemented. |
| 5. Add reference-free notebook metrics | ready_for_review | Notebook uses shared reference-free generation rows instead of raw diagnostics. |
| 6. Add rhythm/grid/strong-beat reference metrics | planned | New reference distribution artifacts and comparison metrics implemented. |
| 7. Wire training and notebook integration | planned | Training generation evaluation and notebook use the same shared metric code. |

## Current Gate

Phase 5 is `ready_for_review`. Do not start Phase 6 until Phase 5 is accepted.

## Phase 1 Log

### Status

complete

### Changed Files

- `docs/generation-evaluation-metrics-plan.md`
- `docs/generation-evaluation-metrics-progress.md`

### Tests Run

- `git diff --check`

### Review Notes

- Saved the approved generation metrics plan in docs.
- Captured the current implementation baseline:
  - training generation evaluation already computes reference-free suite metrics from `SegmentDiagnostics`;
  - current dataset-relative support is limited to figure profile property-rate errors, total relative absolute error,
    and identity total variation distance;
  - notebook metrics duplicate a smaller figure-count comparison and do not yet cover the planned dataset-relative
    distribution families.
- Confirmed the implementation will use `musak_model/configs/analysis/n_grams.yml` instead of a separate metrics
  config.
- Confirmed the implementation will refactor `musak_model/evaluation/generation.py` into a
  `musak_model/evaluation/generation/` subpackage.

## Phase 2 Log

### Status

complete

### Changed Files

- `docs/generation-evaluation-metrics-progress.md`
- `musak_model/evaluation/generation.py`
- `musak_model/evaluation/generation/__init__.py`
- `musak_model/evaluation/generation/evaluator.py`
- `musak_model/evaluation/generation/figure_metrics.py`
- `musak_model/evaluation/generation/protocols.py`
- `musak_model/evaluation/generation/sampling.py`
- `musak_model/evaluation/generation/schema.py`
- `musak_model/evaluation/generation/suite_metrics.py`

### Tests Run

- `git diff --check`
- `uv run python -m py_compile musak_model/evaluation/generation/__init__.py musak_model/evaluation/generation/evaluator.py musak_model/evaluation/generation/sampling.py musak_model/evaluation/generation/suite_metrics.py musak_model/evaluation/generation/figure_metrics.py musak_model/evaluation/generation/protocols.py musak_model/evaluation/generation/schema.py`
- `uv run pytest tests/musak_model/evaluation/test_generation.py`
- `uv run pytest tests/musak_model/training/stages/test_pretraining.py tests/musak_model/training/stages/test_finetuning.py`
- `uv run mypy musak_model/evaluation/generation`
- `uv run python -m py_compile musak_model/evaluation/generation/evaluator.py musak_model/evaluation/generation/suite_metrics.py musak_model/evaluation/generation/figure_metrics.py`

### Review Notes

- Re-read `docs/guidelines.md` and `docs/model.md` before making code changes.
- Converted `musak_model/evaluation/generation.py` into the `musak_model/evaluation/generation/` package.
- Preserved the existing public import path by re-exporting `GenerationSuiteEvaluator` from package `__init__.py`.
- Split the previous module by concern:
  - evaluator orchestration;
  - option/model protocols;
  - sample/report dataclasses;
  - stateless sampling helpers;
  - suite metric aggregation;
  - figure profile metric aggregation.
- Refined the refactor after review:
  - split `_sample` into token-id sampling, model-context checks, logits inference, constraint masking, decoding, and
    sample assembly helpers;
  - split suite metric aggregation into outcome, constraint, bar-completion, activity, token, pitch/rhythm,
    playability, and coordination groups;
  - split figure metric aggregation into generated-artifact construction, comparison metric assembly, and per-sample
    figure counting.
- Did not add new metric behavior in this phase.

## Phase 3 Log

### Status

complete

### Changed Files

- `docs/generation-evaluation-metrics-progress.md`
- `musak_model/configs/analysis/n_grams.yml`
- `musak_model/n_grams/config.py`
- `tests/musak_model/n_grams/test_config.py`

### Tests Run

- `uv run pytest tests/musak_model/n_grams/test_config.py`
- `uv run mypy musak_model/n_grams/config.py`
- `uv run python -m py_compile musak_model/n_grams/config.py`
- `uv run pytest tests/musak_model/n_grams/profile/test_extraction.py tests/musak_model/training/stages/test_figure_profiles.py tests/scripts/test_extract_figures.py`
- `uv run python - <<'PY' ... NGramAnalysisConfig.load() ... PY`

### Review Notes

- Re-read `docs/guidelines.md` and `docs/model.md` before making code changes.
- Kept `musak_model/configs/analysis/n_grams.yml` as the single config source for reference comparison parameters.
- Added explicit canonical config values:
  - `figure_common_mass_threshold: 0.80`;
  - `rhythm_min_n: 2`;
  - `rhythm_max_n: 4`;
  - `grid_alignment_denominators: [1, 2, 4, 8, 16]`;
  - `strong_beat_offsets: [0]`.
- Added typed `NGramAnalysisConfig` fields and validation for rhythm n-gram range, non-empty positive grid denominators,
  non-empty non-negative strong-beat offsets, and common-mass threshold bounds.
- Preserved current figure extraction behavior by leaving existing `min_n` and `max_n` semantics unchanged.
- Added defaults for the new fields so existing focused fixture configs remain valid while the canonical repo config is
  explicit.

## Phase 4 Log

### Status

accepted

### Changed Files

- `docs/generation-evaluation-metrics-plan.md`
- `docs/generation-evaluation-metrics-progress.md`
- `musak_model/n_grams/profile/metrics.py`
- `musak_model/n_grams/profile/metrics/__init__.py`
- `musak_model/n_grams/profile/metrics/distribution.py`
- `musak_model/n_grams/profile/metrics/profile_comparison.py`
- `musak_model/n_grams/profile/metrics/reference_distribution.py`
- `musak_model/n_grams/profile/metrics/stats.py`
- `musak_model/evaluation/generation/figure_metrics.py`
- `musak_model/training/stages/figure_profiles.py`
- `tests/musak_model/n_grams/profile/metrics/test_reference_distribution.py`

### Tests Run

- `uv run pytest tests/musak_model/n_grams/profile/metrics/test_reference_distribution.py tests/musak_model/n_grams/profile/test_builder.py tests/musak_model/n_grams/profile/test_io.py tests/musak_model/evaluation/test_generation.py tests/musak_model/training/stages/test_figure_profiles.py`
- `uv run mypy musak_model/n_grams/profile/metrics musak_model/evaluation/generation/figure_metrics.py musak_model/training/stages/figure_profiles.py tests/musak_model/n_grams/profile/metrics/test_reference_distribution.py`
- `uv run python -m py_compile musak_model/n_grams/profile/metrics/__init__.py musak_model/n_grams/profile/metrics/distribution.py musak_model/n_grams/profile/metrics/profile_comparison.py musak_model/n_grams/profile/metrics/reference_distribution.py musak_model/n_grams/profile/metrics/stats.py musak_model/evaluation/generation/figure_metrics.py musak_model/training/stages/figure_profiles.py tests/musak_model/n_grams/profile/metrics/test_reference_distribution.py`
- searched code, tests, and planning docs for legacy version labels
- `git diff --check`
- `uv run pytest tests/musak_model/n_grams/profile/metrics/reference/test_distribution.py tests/musak_model/n_grams/profile/test_builder.py tests/musak_model/n_grams/profile/test_io.py tests/musak_model/evaluation/test_generation.py tests/musak_model/training/stages/test_figure_profiles.py`

### Review Notes

- Re-read `docs/guidelines.md` and `docs/model.md` before making code changes.
- Split `musak_model.n_grams.profile.metrics` into a subpackage with dedicated modules for shared stats, profile
  property comparison, identity distribution comparison, and reference-distribution comparison.
- Covered common, rare, and novel generated figure mass using the shared analysis config threshold.
- Added distribution-distance comparisons for figure identity, figure properties, contour shape, and duration shape.
- Updated internal callers to import the concrete metric module they use.
- Kept the implementation artifact-oriented and independent from training/notebook integration; wiring remains a later
  phase.

## Next Step

Phase 6: add rhythm/grid/strong-beat reference distribution artifacts and metrics after Phase 5 is accepted.

## Phase 5 Log

### Status

ready_for_review

### Changed Files

- `docs/generation-evaluation-metrics-progress.md`
- `musak_model/evaluation/generation/__init__.py`
- `musak_model/evaluation/generation/reference_free.py`
- `notebooks/model_output_explorer.py`
- `notebooks/utils/__init__.py`
- `notebooks/utils/model_output.py`
- `tests/musak_model/evaluation/generation/test_reference_free.py`
- `tests/notebooks/utils/test_model_output.py`

### Tests Run

- `uv run pytest tests/musak_model/evaluation/generation/test_reference_free.py tests/notebooks/utils/test_model_output.py tests/notebooks/test_model_output_explorer.py tests/musak_model/evaluation/test_generation.py`
- `uv run pytest tests/notebooks/utils/test_model_output.py tests/notebooks/test_model_output_explorer.py`
- `uv run mypy musak_model/evaluation/generation notebooks/utils/model_output.py tests/musak_model/evaluation/generation/test_reference_free.py tests/notebooks/utils/test_model_output.py`
- `uv run mypy notebooks/utils/model_output.py tests/notebooks/utils/test_model_output.py`
- `uv run python -m py_compile musak_model/evaluation/generation/reference_free.py musak_model/evaluation/generation/__init__.py notebooks/utils/model_output.py notebooks/utils/__init__.py notebooks/model_output_explorer.py tests/musak_model/evaluation/generation/test_reference_free.py tests/notebooks/utils/test_model_output.py`
- `uv run python -m py_compile notebooks/model_output_explorer.py notebooks/utils/model_output.py notebooks/utils/__init__.py tests/notebooks/utils/test_model_output.py`
- `git diff --check`

### Review Notes

- Added shared curated reference-free generation metrics from `SegmentDiagnostics`.
- Kept notebook row formatting in `notebooks/utils/model_output.py` so `musak_model` remains notebook-agnostic.
- Updated the model output explorer to show a top-level `Generated Music Summary` table and move detailed raw musical
  diagnostics into the lower-level diagnostics accordion.
- Kept dataset-statistics diagnostic rows unchanged; the new helper is specific to generated model output.
- Removed the generated figure metrics panel, its figure-count CSV browser, and notebook-only figure metric helper code
  because it was slow and not useful in the model output notebook.
- Added a lightweight generated-output-only `Figure Patterns` table below `Generated Music Summary`.
- Added an opt-in reference alignment table for exact figure distribution, contour, duration-shape, property distance,
  common figure mass, rare figure mass, and novel figure mass when a reference figure-count CSV is selected.
- Added generated rhythm-grid rows for onset grid fit, duration grid fit, and strong-beat onset share.
