# Generation Evaluation Metrics Progress

This tracker is the implementation guardrail for `docs/generation-evaluation-metrics-plan.md`. Before changing code in
any phase, re-read `docs/guidelines.md` and keep the phase scoped to the roadmap below.

## Roadmap

| Phase | Status | Review Gate |
| --- | --- | --- |
| 1. Docs and baseline inventory | complete | Plan and tracker saved in docs. |
| 2. Refactor generation evaluation package | complete | Existing behavior preserved; generation package split by concern. |
| 3. Extend shared n-gram analysis config | ready_for_review | `analysis/n_grams.yml` and `NGramAnalysisConfig` own V1 comparison parameters. |
| 4. Add shared V1 figure metrics | planned | Common/rare/novel, property, contour, and duration-shape metrics implemented. |
| 5. Add reference-free notebook metrics | planned | Notebook uses shared reference-free generation rows instead of raw diagnostics. |
| 6. Add rhythm/grid/strong-beat reference metrics | planned | New reference distribution artifacts and comparison metrics implemented. |
| 7. Wire training and notebook integration | planned | Training generation evaluation and notebook use the same shared metric code. |

## Current Gate

Phase 3 is `ready_for_review`. Do not start Phase 4 until Phase 3 is accepted.

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
  - training generation evaluation already computes V0 reference-free suite metrics from `SegmentDiagnostics`;
  - current V1 support is limited to figure profile property-rate errors, total relative absolute error, and identity
    total variation distance;
  - notebook metrics duplicate a smaller figure-count comparison and do not yet cover the planned V1 distribution
    families.
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

ready_for_review

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
- Kept `musak_model/configs/analysis/n_grams.yml` as the single config source for V1 comparison parameters.
- Added explicit canonical config values:
  - `figure_common_mass_threshold: 0.80`;
  - `rhythm_min_n: 2`;
  - `rhythm_max_n: 4`;
  - `grid_alignment_denominators: [1, 2, 4, 8, 16]`;
  - `strong_beat_offsets: ["0"]`.
- Added typed `NGramAnalysisConfig` fields and validation for rhythm n-gram range, non-empty positive grid denominators,
  non-empty non-negative strong-beat offsets, and common-mass threshold bounds.
- Preserved current figure extraction behavior by leaving existing `min_n` and `max_n` semantics unchanged.
- Added defaults for the new fields so existing focused fixture configs remain valid while the canonical repo config is
  explicit.

## Next Step

Phase 4: add shared V1 figure metrics for common/rare/novel mass, figure property distributions, contour distributions,
and duration-shape distributions.
