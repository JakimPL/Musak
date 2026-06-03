# Musicality pass — direction, harmony grounding, phrasing, hands

Status doc for the surface-musicality improvements after the rhythm/breath/motif redesign
([rhythm-redesign.md](rhythm-redesign.md)). Five observed weaknesses, addressed in order; **Phase 1 done**.

## Issues & order
1. Figures wander — no local/global direction.
2. Harmony not respected; chord tones passed indifferently; no sense of bass.
3. No hand relationship (synchronized / conflicting / independent / interleaving).
4. No chords sounded.
5. Melody starts/ends aimlessly; accents have no directional skeleton.

**Steer (user):** harmony grounding, bass, chords, and hand roles must be **learned from the corpus within
the current plan-then-render path — NOT imposed via special per-hand textures/rules** (no hardcoded
`HandTexture` block-chord/bass). Bias *steers, not enforces* triadic tones ([[feedback-inductive-bias-policy]]).

Order: **(1) quick wins → (2) data-learned harmony grounding → (3) phrase direction → (4) hand-coupling axes.**

## Phase 1 — local direction + soft strong-beat harmony (DONE)

### 1a Melodic continuity
The cursor-walk reset every figure to the per-slot register anchor (figures within a slot shared one static
anchor) → no through-line. Now `SurfaceRenderer._fill_hand_bar` threads a **per-hand carried pitch** (the
last emitted note's diatonic position, `anchor + figure_net_contour`); the next figure's anchor is
`_continuity_anchor` = `round(register_anchor + melodic_continuity·(carried − register_anchor))`, and
`target_slope = (register_anchor − anchor) + arch_slope` so the register curve is the **restoring
attractor**. New `RenderConfig.melodic_continuity ∈ [0,1]` (default 0.6; notebook slider). Motif grounded
anchors still take precedence (`_motif_anchor_and_intent`), so restatements reproduce.

### 1b Metrically-weighted soft harmony bias
`figure_selection._chord_tone_coverage` was a flat duration average. `figure_log_scores` now calls
`substitution/scoring.py::harmonic_fit` (metrically weighted via `metrical_weight_over_span` /
`indispensability_per_position`) when given the fired cell's `metrical_position` + `grid_count_per_bar`
(threaded through `select_figure`/`_place_one_figure`); strong beats prefer chord tones, weak beats stay
free. Optional params → falls back to flat coverage for `motif.py::select_motif_seed`. Soft `λ_harmonic`
tilt — **λ=0 inert, figure TV preserved**. (Approximation: `harmonic_fit` treats normalized-duration≈cells
since the base duration isn't fixed at scoring time.)

**Files:** `render/renderer.py`, `render/figure_selection.py`, `render/config.py` + `configs/generation/render.yml`,
`notebooks/utils/form_render.py` + `notebooks/form_render.py`.
**Verified** (exercises, seed 3): strong-beat chord-tone rate 0.89 vs off-beat 0.39; continuity turns a
static pinned figure-sequence (RH span 5) into a connected moving through-line (span 16) with the arch as
attractor. Tests: `test_continuity_anchor_blends_register_and_carried`, `test_melodic_continuity_changes_output`,
`test_strong_beat_sharpens_chord_tone_preference`, `test_metrical_harmony_is_inert_at_zero_lambda`. Gate:
336 synthetic+notebook tests, mypy clean, pylint 10.00, isort clean.

## Measuring output at scale — validation suite + MLflow (DONE)

`make validate-synthetic` (script `scripts/validate_synthetic.py`, config
`configs/generation/validation.yml`, package `musak_model/synthetic/validation/`) renders N exercises per
scale via the form-render path (`render_plan`) and logs generation-quality metrics to MLflow (`make mlflow`
to browse). It **reuses the neural-eval suite** (`evaluation/generation/{suite_metrics,figure_metrics,
rhythm/metrics,musical_metrics}`, `evaluation/diagnostics`) by adapting each rendered `Segment` into a
`GenerationSample` (`validation/adapter.py`); the only generator-specific code is the harness
(`validation/generation.py`), per-scale aggregation (`validation/metrics.py`), the shared MLflow run helper
(`musak_model/mlflow.py::MlflowRun`/`MlflowRunConfig` — one lifecycle/URI/logging implementation that
training, processing, and validation all delegate to), and **new chord-track metrics**
(`validation/synthetic_metrics.py`): `strong/weak_beat_chord_tone_fraction` (+ gap), `mean_abs_melodic_interval`,
`stepwise_fraction`, `leap_fraction`, `chord_onset_fraction`. Fixed-config by default; set
`sweep:` in the config (or `model_copy`) to emit one run per grid cell. Renderer construction was extracted to
`synthetic/render/build.py::build_surface_renderer`; `SyntheticInputs`/`load_synthetic_inputs` moved to
`musak_model/synthetic/inputs.py` (re-exported from the notebook). Run flags: `VALIDATE_FIGURE_DIRECTORY`/
`DATA_DIR`, `VALIDATE_SAMPLES_PER_SCALE`, `VALIDATE_DISABLE_MLFLOW`.

**Baseline (exercises, fitted prior, default knobs)** — the objective the next phases must move:
figure TV-distance ≈ **0.59** (far from corpus), rhythm n-gram TV ≈ **0.52**, **strong-beat chord-tone ≈ 0.54
vs weak ≈ 0.54** (no harmonic advantage at the note level → confirms "harmony not respected"),
`chord_onset_fraction` ≈ **0.016** (→ "no chords"), `mean_abs_melodic_interval` ≈ 2.8 st, rest fraction ≈ 0.20,
in-scale ≈ 0.998, decode/render errors 0. These numbers are the success metric for Phases 2–4.

## Phase 2 — data-learned harmony grounding (emergent bass & chords; NO imposed textures) — PENDING
Chords/bass must **emerge from corpus-learned conditioning**. Directions to design: (a) condition figure
selection on the chord via the already-fit **`p(figure | chord, hand)`** (`synthetic/fitting/figure_by_chord.py`,
used in the retired path, absent from render) as a soft data-learned tilt term; (b) investigate why
chordal (multi-onset) figures never surface (whether `commonness_bias`/the tilt suppress them) and let
per-hand empirical distributions + the low LH home octave carry chord/bass behavior; (c) combine with 1b.
Needs its own design pass.

## Phase 3 — phrase direction & cadential goals — PENDING
Phrase-align the register arch; assert a phrase-start onset; bias the phrase-final fire toward the
cadential resolution implied by `PhraseNode.closing` (tonic/chord-tone landing). Removes aimless start/end.

## Phase 4 — hand-coupling axes (largest, last) — PENDING
The three orthogonal axes from `overview.md §7` — co-activity `h_o`, sync `h_s`, shared harmony — re-homed
onto regions/attacks (`coherence.md` Phase 5), reusing `processes/hand_coupling.py`. Deferred per user.
