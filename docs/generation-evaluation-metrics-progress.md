# Generation Evaluation Metrics Progress

## Phase

Phase 1: Docs and baseline inventory.

## Status

ready_for_review

## Changed Files

- `docs/generation-evaluation-metrics-plan.md`
- `docs/generation-evaluation-metrics-progress.md`

## Tests Run

- `git diff --check`

## Review Notes

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

## Next Step

Phase 2: refactor generation evaluation into a package without behavior changes, preserving existing public imports and
passing existing generation tests unchanged.

