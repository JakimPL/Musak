# Rhythm model redesign — status & hand-over

> Live status doc for the surface-rhythm redesign of the synthetic generator (branch `generator`).
> Read alongside [coherence.md](coherence.md) (overall architecture) and
> [phase-4-motif-variation.md](phase-4-motif-variation.md) (the motif layer this interacts with).
> **As of this writing: R1–R3 done, band-aid reverted, A1–A5 (rest/breath) done, R4–R5 pending.**

## Why this exists (the bug)

`SurfaceRenderer` produced rhythmically incoherent output — some bars half notes, others `1/48`-note
runs. Root cause was **structural**: `slot_base_duration(figure, slot) = slot_duration / figure_span`
*stretched* each figure's normalized rhythm to fill whatever metrical-tree leaf it landed in, and the tree
subdivided each node by an independent coin-flip (arbitrary leaf depths). So note duration =
`leaf_depth × figure_onset_count`, with tuplets when the span didn't divide the leaf in powers of two.

A `FigureNGram` is a **pure normalized rhythm shape** (`builder.py:_normalize_durations` strips the absolute
note value). The absolute value lives in the empirical **`BaseDurationDistribution`**
(`base_durations.parquet`, per `(scale_type, hand, n)`). The fix uses that instead of stretching.

## User constraints (must respect)
- **No rigid pulse**; figures *may* change phrasing tempo, but **in moderation, not randomly** — density
  drifts smoothly.
- **No hardcoded duration ladder** — durations must be **data-derived** (an early attempt with a
  `["1/16",…,"1"]` ladder was rejected as arbitrary/maintenance-heavy). The only durations emitted are ones
  the corpus actually played.
- Don't iterate on line-length nits — the user polishes those. ([[feedback-trivial-formatting]])
- No retraining/regeneration needed: `BaseDurationDistribution` is already in the figure artifacts.

## The model (three parts)
1. **Figures own rhythm (clean base durations).** A figure realizes its normalized shape at an absolute base
   note value from `BaseDurationDistribution`, restricted to bases where every `normalized × base` is a vocab
   duration (`fitting_base_durations`). Emitted via the unchanged `anchor_figure_to_tokens`.
2. **Smooth density process sets the base.** A per-`(scale,hand)` low-frequency offset (in log-duration
   octaves) drifts a target; the chooser snaps it to the **nearest fitting corpus base**. Not i.i.d. per
   figure → moderate drift.
3. **Tree demoted.** It provides the **harmonic frontier** (chord regions = slots), **metrical weights**, and
   **coarse SOUND/REST activity** — never surface note durations. Per-class trees still give restatements a
   shared skeleton.

## Status

| Stage | State | Notes |
|---|---|---|
| R1 RhythmicDensitySampler | **DONE** | `processes/density.py` + `density.yml` + `RHYTHMIC_DENSITY_CONFIG_PATH`; 5 tests |
| R2 clean base-duration chooser | **DONE** | `base_durations.py::fitting_base_durations`/`choose_base_duration`; tests in `test_base_durations.py` |
| band-aid revert | **DONE** | removed `max_figure_onsets`/`allow_tuplets` from `RenderConfig`/`render.yml`/`figure_fits_slot` |
| R3 renderer region-fill | **DONE** | `slots.py` regions + `renderer.py` greedy density fill; **output verified clean** (no tuplet explosion) |
| A1–A5 rest/breath (activity field) | **DONE** | renderer now gates fire/rest via the corpus-fit LGCP activity field; see the section below |
| R4 motif re-integration | **DONE** | per-class texture (shared breath/register/density) + λ_similarity-gated figure-sequence reuse; see the section below |
| R5 density controls + verify | **DONE** | density `amplitude`/`basis_count` exposed through `FormRenderRequest` → notebook sliders; amplitude sweep verified |

**Verified R3 output** (exercises, fitted prior, seed 1): every duration is clean & corpus-attested
(`1/4`, `3/16` dotted-eighth, `1/8`, `1/16`) — **no `1/48`/`1/24`/`1/12`**, density bounded and consistent
across bars. Gate at checkpoint: **28 render tests, 246 synthetic** pass; mypy clean; pylint only line-length.

## Files (new ⊕ / changed △)
- ⊕ `musak_model/synthetic/processes/density.py` — `RhythmicDensityConfig{amplitude,basis_count,decay}`,
  `RhythmicDensitySampler.sample(*, length, rng) -> tuple[float,...]` (octave offsets via `band_limited_random`).
- ⊕ `musak_model/configs/generation/density.yml`; △ `paths.py` (`RHYTHMIC_DENSITY_CONFIG_PATH`).
- △ `musak_model/synthetic/base_durations.py` — `fitting_base_durations(figure, candidates, *, remaining,
  duration_vocabulary)`, `choose_base_duration(figure, candidates, *, density_offset, remaining,
  duration_vocabulary) -> Fraction|None` (snaps `_weighted_median_log(fitting)+offset` to nearest fitting base).
- △ `musak_model/synthetic/render/slots.py` — `render_slots` now returns one `RenderSlot` per **frontier
  node (region)**; `_region_activity(node)` = REST if all leaves REST else SOUND (TIE folded into SOUND).
- △ `musak_model/synthetic/render/renderer.py` — `SurfaceRenderer` gained `base_duration_distribution` +
  `rhythmic_density_sampler` (and keeps `motif_config`, **unused until R4**). `render` walks slot regions;
  `_render_slot` → REST→rest else `_fill_sound_slot` (greedy) → `_place_one_figure` (uses `_fitting_base` =
  `choose_base_duration`) → `_pad_with_rest`. Per-hand density envelopes. **Motif integration removed.**
- △ `musak_model/synthetic/render/config.py` + `render.yml` — band-aid removed; kept tilt defaults
  `lambda_curve=2`, `lambda_harmonic=4`, `lambda_accent=0.5`, `lambda_similarity=0`.
- △ `notebooks/utils/form_render.py` — `_build_renderer` passes the two new fields
  (`inputs.base_duration_distribution`, `RhythmicDensitySampler(RhythmicDensityConfig.load())`).
- △ tests: `processes/test_density.py` (new), `test_base_durations.py` (+R2), `render/test_renderer.py`
  (rewritten for regions; **removed** the motif/tie/coarse-leaf tests; added
  `test_note_durations_stay_on_corpus_base_durations`), `render/test_slots.py` (region assertions).

## Gotchas / invariants
- The motif **modules** (`render/motif.py`, `render/variation.py`, `render/similarity.py`) and their unit
  tests (`test_motif.py`, `test_variation.py`, `test_similarity.py`) are **intact** — only the renderer's
  *integration* was removed. `select_figure(..., intended=…)` still exists.
- The metrical tree still subdivides deeply, but the renderer only uses the **frontier** (regions) +
  weights + activity; deep leaves are ignored except by `_region_activity`. Bounding subdivision to the
  slot level is optional cleanup, not required.
- Construction sites for `SurfaceRenderer` (must pass the two new fields): `test_renderer.py::_renderer`
  and `notebooks/utils/form_render.py::_build_renderer`.
- `harmonic_slot_duration` = slot size = harmonic rhythm (notebook default `1` ⇒ 1 chord/bar ⇒ each bar is
  one region greedy-filled).

## Rest / breath (A-series) — DONE

**Problem:** R3 fixed durations but removed the only silence model — the renderer greedily packed every
SOUND region edge-to-edge, so the melody never breathed. The corpus-fit **LGCP activity field**
(`accent_overrides`, fit by `fitting/accent.py` from `onset_position`+`bar_total` counts) existed in the
artifacts but the R3 renderer never consulted it.

**Fix (mechanism A):** revive that field as a per-hand **fire/rest gate**. Each hand pre-samples a smooth
onset probability `p(t)=expit(β₀+gain·indispensability^γ+envelope)` over a bar-aligned grid, draws a
Bernoulli onset mask (`processes/accent.py::draw_onset_mask`), and the renderer walks a position cursor
per `(bar, hand)`: non-firing cells (or REST regions) → rests that cluster into breath via the smooth
envelope; firing cells → a figure placed exactly as R3 (clean density base). The firing cell's `p(t)`
also replaces the static `slot.weight` in figure selection (its documented "double role"). The tree's
SOUND/REST region acts as the coarse gate ANDed with the onset mask (`onset and gate`, like the retired
generator). Phrase-end breath and hand co-activity/sync are deferred (see decisions in the plan).

**Grid:** the field is gated at the **fit grid denominator**, not the finest note. `grid_denominator` is
now **persisted in `FittedGeneratorConfig`** (default 4 = `FIT_GRID_DENOMINATOR`) so the render grid
matches the fit (`grid_count_per_bar = bar_duration · grid_denominator`, `cell_duration = 1/grid_denominator`);
the fitted `baseline_logit`/`metric_gain` are calibrated for that grid's indispensability levels.

**Files:** ⊕ `processes/accent.py::draw_onset_mask`; △ `render/renderer.py` (`SurfaceRenderer` gained
`accent_field_sampler` + `grid_denominator`; `render` samples per-hand accent weights + onset masks;
`_fill_sound_slot`/`_pad_with_rest`/`_emit_or_rest`/`_render_slot` replaced by `_fill_hand_bar` +
`_next_fired_position` + `_slot_index_at` + `_feasible_entries` + `_emit_rest`); △ `fitting/artifacts.py`
(`grid_denominator` field + `DEFAULT_GRID_DENOMINATOR`), `fitting/fit.py` (thread it through); △ build
sites `notebooks/utils/form_render.py::_build_renderer` and `test_renderer.py::_renderer`. Construction
sites for `SurfaceRenderer` now pass `accent_field_sampler` + `grid_denominator`.

**Verified** (exercises fitted prior, seed 1, default config): rest fraction ≈ 0.31, rests clustered
(e.g. a whole-bar `r1` in the LH while the RH plays; `r3/8 r1/16` runs), hands breathe independently,
all durations clean/corpus-attested. Tests: `processes/test_accent.py` (mask determinism/density/
clustering), `render/test_renderer.py` (`test_render_emits_rests_with_default_config`,
`test_lower_baseline_logit_increases_rests`; clean-durations + determinism kept). Gate: 330
synthetic+notebook tests pass; mypy clean; pylint 10.00; isort clean.

## R4 — motif re-integration (cursor-walk) — DONE

Two coordinated parts, both keyed by `(class_label, bar_span)` like `class_metrical_tree`:

**Per-class texture (always on — structural repetition).** Register anchors, density offsets, accent
weights and the onset mask are sampled **once per class** and reused for every same-class segment, then
concatenated in segment order into the global per-hand arrays (`_assemble_textures` /
`_sample_class_texture`). So restatements share the same breath/register/density skeleton — the structural
meaning of the FormTree's repetition. This changed the A-series sampling from one global draw to per-class
draws (determinism per seed preserved; figure marginal unchanged).

**Motif figure-sequence reuse (gated on `lambda_similarity > 0`).** The renderer loop is now
**segment-major**. For each segment a per-(class,hand) `_MotifWalk` is created: a FRESH class **records**
the figures it places (as a `MotifSchema` of `MotifFigure`, indexed by fire ordinal, with
`anchor_offset = anchor − base_anchor`); a SAME/VARIANT restatement looks up the class's schema, applies
`vary_motif` (IDENTITY/transpose for SAME, invert/retrograde/transpose for VARIANT) and `ground_motif` to
the instance's `base_anchor`, then at the k-th fire feeds the grounded figure as `intended` into
`select_figure` (and overrides the anchor). Figures are still emitted at **density-governed bases**. At
`λ_similarity = 0` no `_MotifWalk` is built → `intended` is `None`, no anchor override → identical to the
base path (figure marginal / TV preserved).

**Files:** △ `render/renderer.py` (`_MotifWalk`/`_ClassTexture`; segment-major loop; `_assemble_textures`,
`_sample_class_texture`, `_segment_motif_walks`, `_motif_anchor_and_intent`, `_record_motif_figure`;
`render` split into `render`/`render_plan` returning `RenderResult(segment, chords)`); reuses
`render/motif.py` (`MotifSchema`/`ground_motif`) + `render/variation.py` (`vary_motif`) +
`figure_selection.select_figure(intended=…)` unchanged. Tests: `test_repeats_reuse_motif_at_high_similarity`
(exact reproduction of a SAME restatement at high λ), `test_similarity_changes_output_for_repeats` (gate).

**Verified** (exercises fitted prior): a form `[0,1,2,1,3]` restates class 1 as VARIANT at λ=8 — material
reused transformed, decode-clean. Gate: 332 synthetic+notebook tests; mypy clean; pylint 10.00; isort clean.

## R5 — density controls + verify — DONE
- `FormRenderRequest` gained `density_amplitude` / `density_basis_count`; `render_form_segment` builds a
  `RhythmicDensityConfig` from them and passes it to `_build_renderer` (no longer a fixed
  `RhythmicDensityConfig.load()`); the notebook exposes an amplitude slider + an oscillation-count input
  under a "Rhythm" section.
- **Verified** (exercises fitted prior, seed 1, 8 bars): amplitude `0.0` → durations cluster tightly
  (50/73 eighths, mean 0.147); `1.0` → wider (27×1/16 + 30×1/8 + some 1/4,1/2); `3.0` → widest spread
  (more 1/16 *and* more 1/2, mean 0.165). All decode-clean, all corpus-attested durations. The knob
  controls how much the phrasing tempo drifts — smooth and moderate, not random.

## Verification commands
```
uv run pytest tests/musak_model/synthetic -q
uv run mypy musak_model/synthetic/render musak_model/synthetic/processes/accent.py musak_model/synthetic/processes/density.py musak_model/synthetic/base_durations.py musak_model/synthetic/fitting
uv run pylint musak_model/synthetic/render musak_model/synthetic/processes/density.py
make notebook-form_render   # select <encoded>/figure/all ; Render
```
Duration-dump (the coherence check): render via `notebooks.utils.render_form_segment` on the `exercises`
fitted prior and group `extract_hand_onset_runs` onset durations per bar — all should be clean vocab values.
