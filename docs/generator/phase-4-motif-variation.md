# Phase 4 — Motif reuse-with-variation (the surface-coherence bridge)

> Concrete implementation plan for Phase 4 ([coherence.md](coherence.md) §14, §16.2 steps 2/4/5). Phases 0–3 are
> in place: a `FormPrior` samples a `FormTree` (phrases + per-phrase cadences + repetition-class segments), the
> harmony grammar roots per phrase, and `SurfaceRenderer` fills the metrical slots. Today the renderer samples
> **one** metrical tree for the whole piece and draws each slot's figure independently — so the `FormTree`'s
> `segments`/repetition classes are inert. Phase 4 makes them audible: a repetition class is rendered **once**
> into a reusable motif, and its restatements reuse that motif under **variation operators**, re-grounded to the
> new anchor/slots/chords. This is what turns "segment B restates A" into recognizable-but-not-identical material.

## Governing decisions (committed; defensible against §14/§16.2)

1. **Per-class metrical tree, shared by both hands.** The metrical tree is a property of a repetition **class**
   (§16.2 Fix 3), sampled once and reused by restatements. We keep the Phase-2 both-hands-share-one-tree
   simplification (per-hand trees stay Phase 5), so the tree is per **(class_label, bar_span)**: reused when a
   restatement has the same span, freshly sampled otherwise (FormSampler picks classes independent of span, so
   span-matched reuse is the common period/sentence case; span mismatch degrades gracefully to a fresh tree).
2. **Motif is per (class, hand).** A hand's line within a class is its motif; each `(class_label, hand)` gets one
   seed `MotifSchema`. This composes with the current "both hands render from the shared slots, each with its own
   register anchor + hand-filtered figure vocabulary" structure.
3. **Operators split by whether they preserve the slot grid.** Grid-preserving operators
   (`IDENTITY`, `DIATONIC_TRANSPOSE`, `INVERT`, `RETROGRADE`) re-ground trivially (same slots, re-anchor,
   re-score). Grid-changing operators (`ORNAMENT`, `AUGMENT`/`DIMINISH`, `FRAGMENT`) need slot re-fitting and fall
   back to `IDENTITY` on failure (§14.6). `SEQUENCE`/`TRUNCATE`/`EXTEND` are compositions deferred to a follow-up.
4. **`similarity_fit` is an additive tilt term**, `λ_similarity · similarity_fit`, with `λ_similarity = 0`
   recovering today's corpus-marginal behaviour exactly — the TV-distance calibration is preserved by
   construction (§14.5). Re-grounding never emits the operator's "intended" figure verbatim; it draws a real
   vocabulary figure near it.
5. **Determinism preserved.** All sampling threads the single `rng`; per-class caches are keyed deterministically;
   seed-selection chooses among i.i.d. renders (not autoregressive — §14.2).

## Data structures (`synthetic/render/motif.py`, frozen)

```
MotifFigure:   slot_index: int            # index into the class tree's SOUND slots
               figure: FigureNGram         # relative contour + normalized rhythm
               anchor_offset: int          # diatonic steps from the motif's first anchor
MotifSchema:   figures: tuple[MotifFigure, ...]   # one per SOUND slot, in order
               sound_slot_count: int
MotifInstance: schema: MotifSchema; base_anchor: int   # grounding = base_anchor + slot chords (held externally)
```
A `MotifSchema` is anchor- and time-relative (the rhythm lives in the shared class tree), so it transposes/retimes
for free. Reuse = transform the schema, then re-ground onto the instance's slots/anchor/chords.

## Batches

**A — Per-class metrical trees (foundational).** `class_metrical_tree(metrical_sampler, form, *, time_numerator,
time_denominator, rng)`: sample one tree per `(class_label, bar_span)`, translate its bars to each segment's
`start_bar`, concatenate into the piece tree (`_translate_node` rebuilds nodes at shifted offsets). `render` uses
it instead of the single whole-piece sample. *Verify:* restated same-class/span segments have identical leaf
structure modulo offset; existing renderer tests still pass (they assert properties, not exact tokens);
determinism holds.

**B — MotifSchema + seed selection.** For each fresh `(class, hand)`, render the class's first instance `N_seed`
times (i.i.d. figure draws over its SOUND slots), score each by `Q` = weighted sum of mean metrically-weighted
`harmonic_fit` + contour smoothness (penalize large inter-figure anchor jumps) + figure typicality (count^β), keep
the best, extract its `MotifSchema`. *Verify:* seed selection is deterministic per seed; higher-Q seed chosen;
`MotifSchema` round-trips (extract → re-render identity → same figures).

**C — Variation operators (`synthetic/render/variation.py`).** `MotifSchema → MotifSchema`. Grid-preserving core +
grid-changing with fallback (decision 3). `variation_operator(kind, schema, *, rng, scale_type, vocabulary)` and an
operator sampler keyed by `SAME`/`VARIANT` + `variation_budget` (`SAME → {IDENTITY, DIATONIC_TRANSPOSE}`;
`VARIANT →` a budget-scaled composition). *Verify:* each operator's invariants (DIATONIC_TRANSPOSE shifts every
anchor by Δ, preserves rhythm; INVERT negates relative steps; RETROGRADE reverses; ORNAMENT raises n; FRAGMENT is a
contiguous subsequence); composition associativity; fallback when a grid-changing op can't fit.

**D — Re-grounding + `similarity_fit`.** `select_figure` gains an optional `intended: FigureNGram | None` +
`lambda_similarity`; `similarity_fit(candidate) = -figure_edit_distance(candidate, intended)` over the vocabulary
neighborhood (figures within edit-distance ε in `(degree-steps, normalized-duration)` space), snapping to the
nearest real figures when `intended` is out of vocabulary. `figure_edit_distance` is a Levenshtein over the onset
sequence (substitution cost = |Δdegree| + duration mismatch). Re-grounding: place the varied schema's figures on
the instance's slots with the instance's register anchor (+ `DIATONIC_TRANSPOSE`), each scored against the
instance's chord. *Verify:* `λ_similarity = 0` reproduces the Batch-A figure marginal exactly (TV unchanged);
`λ_similarity → large` returns the intended figure when it is in vocabulary.

**E — Integration + config + notebook.** `SurfaceRenderer.render` consumes `FormTree.segments`: per fresh class
→ per-class tree + per-(class,hand) seed motif; per restating instance → reuse schema under a sampled operator,
re-grounded. New config `motif.yml` (`seed_candidate_count`, quality weights, `variation_budget`,
`lambda_similarity`, `edit_distance_radius`) + `MotifConfig`; thread `variation_budget` from a control. Update
`notebooks/form_render.py` to surface motif reuse (e.g. show which segments restate which class). *Verify:*
restatements recognizable but not identical; repeated-figure-family & variation-after-repeat rates move with
`variation_budget`; **figure TV at `λ_similarity = 0` matches the pre-Phase-4 marginal**; the 8-bar period still
reproduces; full gate green.

## Conventions & limits
Frozen pydantic for serialized/config, frozen dataclasses for internal state, verbose names, `Final` constants,
reuse existing scoring (`slope_fit`/`harmonic_fit`/`accent_fit`/`onset_chord_tone_fraction`) and the figure
vocabulary; `uv run` tooling; minimal docs. Carries §14.6's limits: render-once propagation (mitigated by
seed-selection), orbit coverage = operator-set, edit-distance perceptual flatness, within-class only (no
cross-class thematic derivation). Per-hand trees and `SEQUENCE`/`TRUNCATE`/`EXTEND` deferred.
