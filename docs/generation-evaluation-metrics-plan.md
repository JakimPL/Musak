# Generation Evaluation Metrics Plan

This plan is governed by `docs/guidelines.md`. Implementation must keep code modular, typed, scoped to the owning
packages, and tested under matching test paths.

## Goals

- Improve the generation metrics shown below model output in `notebooks/model_output_explorer.py`.
- Reuse the same generation evaluation metric code in training and the notebook.
- Keep `musak_model/configs/analysis/n_grams.yml` as the single source of truth for n-gram and distribution-comparison
  parameters.
- Preserve n-gram comparison, but make it align with the V1 dataset-relative distribution goals in `docs/metrics.md`.
- Implement incrementally so each phase can be reviewed before the next one starts.

## Current State

`musak_model/evaluation/generation.py` currently owns training-time generation sampling and metrics. It already logs
many V0 reference-free metrics from `SegmentDiagnostics`, including decode/end/constraint rates, bar completion,
silence/activity, token fractions, tonality, density, playability, and hand coordination.

The current V1 slice is limited. Training generation evaluation compares generated figure artifacts against processed
dataset figure artifacts with:

- figure profile property-rate errors: monophonic, chords-only, and in-scale rates;
- figure total relative absolute error;
- figure identity total variation distance.

The notebook implementation in `notebooks/utils/model_output.py` duplicates part of this logic and shows a smaller
table: generated/reference group counts, total occurrences, unique figure counts, comparable groups, mean identity
total variation distance, and mean total relative absolute error. It does not yet expose common/rare/novel figure mass,
contour distributions, duration-shape distributions, rhythmic n-grams, grid-alignment distributions, duration entropy,
or strong-beat onset fractions.

## Architecture

Refactor `musak_model/evaluation/generation.py` into a subpackage:

```text
musak_model/evaluation/generation/
  __init__.py
  evaluator.py
  sampling.py
  suite_metrics.py
  reference_free.py
  figure_metrics.py
  rhythm_metrics.py
  notebook_rows.py
```

Responsibilities:

- `evaluator.py`: `GenerationSuiteEvaluator` orchestration and public evaluator types.
- `sampling.py`: autoregressive sampling helpers, constraint reporting, and segment construction.
- `suite_metrics.py`: suite-level aggregation and MLflow metric-name assembly.
- `reference_free.py`: curated reference-free generation summaries shared by training and notebook code.
- `figure_metrics.py`: V1 figure comparisons from existing figure count/profile artifacts.
- `rhythm_metrics.py`: rhythm, duration, grid-alignment, and strong-beat distribution extraction/comparison.
- `notebook_rows.py`: small formatting adapters that turn shared metric records into notebook table rows.

The package `__init__.py` should re-export the current public evaluator symbols so existing imports from
`musak_model.evaluation.generation` remain valid.

## Shared Analysis Config

Do not add a separate generation metrics config. Extend `NGramAnalysisConfig` and
`musak_model/configs/analysis/n_grams.yml` so the same parameters drive dataset processing, train/validation metrics,
training generation evaluation, and notebook comparison.

Add comparison settings to the existing analysis config, including:

- `figure_common_mass_threshold`, default `0.80`;
- `rhythm_min_n` and `rhythm_max_n`;
- `grid_alignment_denominators`;
- `strong_beat_offsets`, with a default policy derived from measure start when no explicit values are configured.

The implementation should keep existing figure `min_n` and `max_n` behavior intact. If naming evolves during
implementation, keep the config shape explicit, Pydantic-validated, and documented by tests.

## V1 Metrics

Implement these dataset-relative metrics as shared code:

- common, rare, and novel figure mass;
- figure identity distribution distance;
- figure property distributions;
- contour distribution distance;
- duration-shape distribution distance;
- rhythmic n-gram distribution distance;
- duration entropy and duration-value distributions;
- grid-alignment distributions;
- strong-beat onset fraction.

Common figures are the smallest set of reference figures covering the configured `figure_common_mass_threshold` within
each compatible slice. Rare figures are present in the reference slice but outside that common set. Novel figures are
absent from the reference slice.

Use compatible metadata slices where available:

- figure metrics: scale type, hand, and n-gram length;
- rhythm/grid/strong-beat metrics: scale type, time signature, optional hand, and n-gram length where applicable.

Existing `figure/all/counts.csv` remains valid for figure-only comparison. Add richer reference-distribution artifacts
beside the existing figure artifacts only when needed for rhythm/grid/strong-beat comparisons.

## Notebook Behavior

Update `model_output_explorer.py` to call shared generation evaluation helpers rather than notebook-only duplicate
metric code.

Below output, show two tables:

- `Reference-Free Metrics`: a compact curated set from shared reference-free generation summaries, not the current raw
  diagnostics list.
- `Dataset-Relative N-Gram Metrics`: n-gram and distribution comparison metrics. When only `figure/all/counts.csv` is
  available, show figure-only rows and omit richer rhythm/reference rows.

Keep detailed diagnostics/debug tables available only in lower-level debug sections if still useful.

## Incremental Phases

1. Docs and baseline inventory.
2. Refactor generation evaluation into a package without behavior changes.
3. Extend `analysis/n_grams.yml` and `NGramAnalysisConfig`.
4. Add shared V1 figure metrics.
5. Replace notebook raw diagnostics with reference-free shared rows.
6. Add rhythm/grid/strong-beat reference distribution artifacts and metrics.
7. Wire all shared metrics into training generation evaluation and notebook display.

Each phase must update `docs/generation-evaluation-metrics-progress.md` before stopping for review.

## Test Strategy

- Preserve existing generation evaluation tests through the package refactor.
- Add config validation tests for `NGramAnalysisConfig`.
- Add focused unit tests for common/rare/novel mass, contour extraction, duration-shape extraction, and figure property
  distribution comparisons.
- Add focused tests for rhythmic n-gram, duration entropy, grid alignment, and strong-beat onset extraction from
  tokenized segments.
- Add notebook row-helper tests proving the notebook uses shared generation evaluation code.
- Run existing generation, figure-profile, notebook model-output, and n-gram tests after each behavioral phase.

