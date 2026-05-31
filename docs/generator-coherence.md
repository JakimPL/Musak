# Design — Stochastic internal coherence: a top-down "plan-then-render" generator

> Status: **high-level design** (deliberation). Destined for `docs/generator-coherence.md`.
> Scope locked: short exercises (**4–16 bars**), **single key** (modulation deferred behind a switch),
> classical/stochastic (non-neural) throughout. Specifies the generative model — *what is sampled, conditioned
> on what* — not the implementation.

## 1. Problem (one sentence)

The current generator decides everything at the level of **grid cells and tokens**, cell-by-cell, from low-order
global processes sampled once and a greedy left-to-right figure walk; it has no object that carries **form,
phrase, repetition, functional harmony, or cadential closure**, so its output is structurally aimless. The cure:
make those higher objects the model's *primary entities*, sample them **top-down**, render tokens only at the
leaves.

## 2. Governing principle

**Plan top-down; render bottom-up; never condition on previous output.** A piece is a *derivation* of nested
trees, sampled before any note exists. A leaf's content depends on its **ancestors in the tree** — not on the
preceding note. Coherence is *shared ancestry*; the autoregressive framing (the diagnosed source of aimlessness)
never enters. Hand-coded structure is minimised: where a low-bias **data-derived** prior is cheap and robust we
learn it; we hand-specify only rule *sets* that are the grammar of the idiom (not stylistic taxonomies) and fit
their *probabilities*.

**Inductive-bias policy.** Provide inductive bias sparingly and, wherever possible, *learn it from the corpus*
rather than impose arbitrary Western-European music-theory categories (much of music theory is unprincipled
convention). The guiding tension: every classical construct we bake in (the tonic/subdominant/dominant functional
division, named cadences, diatonic-triad realization) narrows the model toward the rigid tonal idiom and away from
arbitrary music — so prefer to *measure* a phenomenon from data over *naming* it, and keep any classical construct
**isolated and swappable**. We accept such constructs **only where the target — classical sight-reading exercises
— justifies them**. The T/S/D division is itself over-constraining in a broad musical sense and is justified here
only by the target repertoire.

## 3. Methods and what each supplies (sources at end)

| Need (user point) | Method | Status in this design |
|---|---|---|
| Repetition / phrase parallelism, low-bias (2,4,7) | **Empirical repetition-structure prior** (metric-grid segmentation + figure-similarity + corpus histogram of repetition strings); SSM / SIATEC-COSIATEC as optional refinement | **Learned from data** (cheap, safe) — replaces named-template form |
| Functional, recursive harmony; tonic open/close (1,2,7) | **Generative Syntax Model** (compact PCFG of tonal harmony), probabilities fit by **inside-outside + Bayesian smoothing** | **Rule set specified (the idiom), probabilities learned** |
| Plan rhythm top-down; whole/held notes; multi-bar spans; structured rests (3,5,6) | **Probabilistic metrical / rhythm trees** | Hand-specified split grammar (meter-derived), probabilities tunable/fittable |
| Harmony & meter share ONE skeleton, "no ifs" (6) | **GTTM time-span reduction** | structural — harmony hangs on the metrical tree |
| Repetition-with-variation; "similar not identical, same rhythm, new anchor" (2,4) | **Cope recombinance** (render-once-then-vary) + transformation operators; edit/feature distance for retrieval | render-once now; abstract schema deferred |
| Openings/closure (7) | **Positional opening/closing harmonic statistics** + grammar cadence head | soft data-derived bias + grammar |

## 4. The generative model, layer by layer

Each layer defines `p(layer | parents)`; layers are sampled in order and **frozen** before the next.

### Inputs (the request — controllable style API, preserved)
Scale (root, type), meter `(n/d)`, `bar_count ∈ [4,16]`, style knobs: density, register home/spread per hand,
hand co-activity & sync, **harmonic-rhythm rate**, **variation budget**.

### Layer 1 — Form  →  `FormTree`  (LEARNED, low-bias — no named templates)
We do **not** impose period/sentence. Form is a **partition into metric-grid segments + a parallelism/repeat
relation**, sampled from a corpus-estimated distribution.
- **Estimation (offline):** segment each corpus piece on the metric grid into a base unit (e.g. 1–2 bars); score
  segment-to-segment similarity from the existing figure n-grams (figure overlap / template edit distance);
  threshold into `SAME / VARIANT / DIFFERENT`; reduce each piece to a **canonical repetition string** (`A A′ B
  A″`, first occurrence = new letter, near-duplicate = prime); **histogram strings per `bar_count`** (back off to
  coarser "repeat-or-not per segment" when counts are sparse).
- **Sampling (generation):** draw a repetition string `~ P(structure | bar_count)`. It fixes which segments are
  restatements/variants of which → the `restates` links that carry coherence. No stylistic taxonomy is asserted.
- **Openings/closings:** separately estimate **positional harmonic statistics** — the harmonic function / figure
  family that tend to start and end pieces/segments — and use them as a *soft bias* on the first segment (tonic
  establishment) and last segment (cadential close). Honest and low-bias: a tendency, not an enforced `if`.
- Output: `FormTree` = ordered `SegmentNode`s with `(span, restates?, variation_budget)` + opening/closing bias.
- *(Named period/sentence templates and automatic phrase-boundary detection are a later refinement / validation
  lens, not the generative backbone — see §7 L1.)*

### Layer 2 — Metrical / time-span skeleton  →  `MetricalTree`  (the SHARED skeleton)
Per segment, sample a time-span tree by a recursive **subdivision grammar** `G_meter`, top-down:
- A bar node in meter `(n/d)` splits by **meter-determined** groupings (4/4→`{2×½}`; 3/4→`{3×¼|¼+½|½+¼}`;
  6/8→`{2×⅜|3×¼}`), each with a probability; sub-bar nodes split binary/ternary per the remaining duration. A node
  **terminates** with probability rising as it gets short/weak; the leaf is typed `SOUND / TIE / REST` by a
  categorical on the node's metrical weight and the density knob.
- **Metrical weight = tree rank** (GTTM), **replacing** `gcd(k,M)/M`; gives a partial order of strength directly
  and handles compound/asymmetric meters.
- **Whole/held notes & multi-bar spans are structural** (a bar node terminating as one `SOUND` = whole note; a
  `TIE` across a barline = held note). **Rests are a leaf type in the same grammar** (point 6).
- **Shared down to the harmonic-rhythm level (adaptive, sampled per piece):** nodes at/above that level form the
  skeleton harmony hangs on; below it, each hand subdivides **independently** for surface rhythm.

### Layer 3 — Harmony  →  `HarmonyTree`  (compact single-key GSM, hung on the skeleton)
Derive a functional tree by a **pruned** `G_harm`, region boundaries **hard-aligned** to skeleton nodes at/above
the harmonic-rhythm level; recursion bounded to fill exactly that many surface-chord slots:
- Head `Segment → T`; prolong `T → T T`; prepare `T → D T`, `D → S D`; instantiate to scale-degree chords by
  pruned substitution `T→{I,vi}`, `D→{V,V7,vii°}`, `S→{IV,ii,ii6}`.
- `cadence_target` (from Layer 1's closing bias) constrains the final terms (`PAC ⇒ …D→T`, root-position /
  tonic-soprano flagged for the voicer; `HC ⇒ end on D`).
- Leaves = **preserved** key-relative `Chord` objects (`harmony/schema.py`, `expand_chord_to_tones`).
- **Probabilities are LEARNED:** inside-outside / EM on decoded corpus harmony, with **Dirichlet smoothing** and
  theory-initialised priors (local-optimum mitigation). The **rule set is fixed** (the tonal idiom, low-bias);
  the rule *set* is **not** induced (that's fragile — §7 L1). Modulation rules are a deferred switch.

### Layer 4 — Motif plan  →  `MotifPlan`  (render-once-then-vary)
- A **motif** = an anchored, time- and chord-grounded **sequence of figure n-grams** (the user's definition).
- The **first** occurrence of a repetition-class (label `A`) is rendered **concretely** via the Layer-5 surface
  (given its slot's anchor/chord/metrical context); **sample a few candidate seeds and keep the best** by a
  quality score (harmonic fit + contour smoothness) — that realization becomes `prototype_A`.
- **Subsequent occurrences** of `A`/`A′` reuse `prototype_A` under a **`VariationOperator`** drawn per the
  repetition relation (`SAME → IDENTITY / DIATONIC_TRANSPOSE` to the new slot's anchor; `VARIANT → ORNAMENT,
  AUGMENT/DIMINISH, INVERT, FRAGMENT, SEQUENCE`, concentration set by `variation_budget`). The operator
  **re-grounds** the motif to the new anchor/chord; harmonic coherence under the new chord is restored by the
  surface `harmonic_fit` tilt. "Same rhythm, new anchor" *is* DIATONIC_TRANSPOSE.
- *Abstract idealised motif schemas are deferred (§7 L3); storing motifs as anchored figure-n-gram sequences
  keeps that door open.*

### Layer 5 — Surface render  →  `EventStream`  (the PRESERVED I-projection)
For each surface leaf (per hand) with anchor, span→`base_duration`, chord, and `(prototype, operator)`:
1. Apply the operator to the prototype in template space → an **intended** relative figure (for free material,
   no prototype).
2. Draw the concrete figure from the empirical group `(scale, hand, n)` by the tilt
   `p(f) ∝ c(f)^β · exp(λ_curve·slope_fit + λ_harmonic·harmonic_fit + λ_accent·accent_fit + λ_similarity·similarity_fit)`,
   `similarity_fit(f) = −dist(f, intended)` over the edit-distance neighborhood (`λ_similarity=0` + no prototype
   recovers today's behaviour).
3. **`base_duration` is read from the metrical leaf's span** (figure rhythm scaled to the slot), **not**
   fit-to-remaining-bar — removing the compression bias (point 3). `TIE`/`REST` leaves render directly.
4. Encode → tokens; gate through `GenerationConstraintState` (lone hard gate); resample the figure within the
   slot on rejection; escalate fallback (simpler figure → single note → rest).

**Calibration survives:** at `λ_similarity=0` with no prototype the figure marginal = corpus marginal, so the TV
objective (`figure_distribution_metrics`) holds; each `λ` is a stability↔fidelity dial.

### Layer 6 — Encoding  →  `Token[]`
Deterministic serialization. Tokens are **only the output format** (point 8), never a working entity.

### Cross-cutting — two hands
Share `FormTree`, `HarmonyTree`, and the skeleton at/above the harmonic-rhythm level; each gets its own surface
subdivision + motif realization below; re-coupled by the existing `hand_coupling` (co-activity `h_o`, sync
`h_s`), now acting on regions/attacks rather than per-cell.

## 5. Keep / demote / replace

| Current module | Verdict |
|---|---|
| `processes/pitch.py` (register curve) | **Demote → render prior** on anchor + slope bias. |
| `processes/accent.py` (LGCP, `gcd(k,M)/M`) | **Demote → soft accent/density prior**; structural role → MetricalTree; indispensability → tree rank. |
| `processes/chord_track.py` (1st-order Markov) | **Replace → HarmonyTree (GSM, learned probs)**; keep the `Chord` representation. |
| `processes/hand_coupling.py` | **Keep, re-home** (regions/attacks). |
| `substitution/generator.py` greedy walk + `_fitting_base_durations` | **Replace → top-down renderer**; base-duration from leaf span. |
| `substitution/sampling.py`+`scoring.py` | **Keep, extend** with `similarity_fit`; `harmonic_fit` reads grammar chord. |
| `substitution/chord_figure.py` (`p(figure\|chord)`) | **Keep**. |
| `substitution/texture.py` | **Keep, re-home** as per-segment/hand render choice. |
| Figure vocabulary + `commonness_bias`; `figure_distribution_metrics` | **Keep — canonical surface + objective.** |
| `generation/constraints.py` | **Keep verbatim — only hard gate.** |
| Corpus chord decoder (Viterbi, `fitting/chord.py`) | **Keep for `p(figure\|chord)` + as the labeled input to inside-outside; grammar-parser upgrade later.** |

## 6. Assumptions (explicit)
- **A1** Hierarchical organization: form ⊃ segment ⊃ time-span ⊃ figure ⊃ note.
- **A2** Harmony is functional and approximately context-free (derivations, not Markov transitions); the rule set
  is the idiom (low-bias), only probabilities are learned.
- **A3** Rhythm is meter-aligned recursive subdivision; metrical strength = tree rank.
- **A4** Coherence = parallelism: repetition-with-variation of a tiny motif set, structured by a **data-derived**
  form prior.
- **A5** Local figure surface statistics are well-captured by the empirical FigureNGram vocabulary and preserved (TV objective).
- **A6** The plan is sampled top-down and **not** revised by render feedback (only local figure resampling).
- **A7** Single key; short pieces ⇒ metric-grid segmentation is reliable; modulation deferred.

## 7. Limitations (honest)
- **L1 — Structure induction is bounded.** We deliberately *avoid* the fragile parts: inducing the harmony rule
  *set* unsupervised is data-hungry/fragile (jazz SoTA F1 0.49 vs 0.64 supervised), and automatic phrase-boundary
  detection caps at ~0.6 F1 and is weak on short smooth lines — so we use a **fixed** harmony rule set (learn only
  probabilities) and the **metric grid** for segmentation. The data-derived form prior inherits noise from the
  segment-similarity threshold and sparse histograms on a small corpus (mitigated by back-off). Adaptor/fragment
  grammars (learning reusable subtrees) are conceptually ideal but have **no proven music application** — a
  research bet, deferred.
- **L2 — Rhythm tree.** Strict meter-alignment under-represents syncopation/cross-accent (only via low-probability
  `TIE` displacement); tuplets need explicit split rules.
- **L3 — Variation.** Render-once-then-vary **propagates the seed realization's quality** to its restatements
  (mitigated by seed-selection). Operator-orbit reaches only **operator-images** of a prototype; **edit distance
  is perceptually flat** (equal weight per edit, ignores salience); composing operators can drift perceptually far
  while staying close in orbit count; the snap-to-vocabulary step trades coherence against TV-fidelity (the
  `λ_similarity` dial). An **abstract motif schema** (idealised contour/rhythm) would decouple "good idea" from
  "good realization" but is **unsolved even with deep learning** — deferred; a property-vector/learned similarity
  metric is a later validation/retrieval upgrade that doesn't change the generative story.
- **L4 — No plan feedback.** A slot hard to render under constraints can only locally degrade, not re-plan.
- **L5 — Calibration scope.** TV-distance guarantees near-reference *figure marginals* only; structural
  plausibility has no corpus-calibrated guarantee until the §8 structural metrics + fitted grammar probabilities
  are in place. The grammar makes structure *valid*, not provably *distributionally matched*.
- **L6 — Scope.** Tied to short, single-key pieces; longer/multi-section/modulating music needs a richer form
  prior + the modulation switch.

## 8. How we'll know it works (when this becomes an implementation plan)
Structural metrics (`model-improvement-roadmap.md`): repeated-figure-family rate, variation-after-repeat rate,
phrase-begin/end stability, decoded chord-transition plausibility, harmonic consonance at coincident onsets; plus
the preserved figure TV-distance (must match the corpus at `λ→0`); and human inspection (notation / piano roll /
audio).

## 9. Decisions locked
- **Form:** learned low-bias **repetition-structure prior** (metric-grid segmentation + figure-similarity +
  corpus histogram); named templates deferred. **Minimal 2-level grouping (§15 Fix 1):** parallelism (`restates`)
  at the segment level, harmony-rooting + cadence at the phrase level; phrase-length & per-position
  `P(closing pattern | phrase position)` learned from the corpus (§19).
- **Harmony:** **compact single-key GSM rule set fixed; probabilities learned** (inside-outside + Dirichlet).
  Modulation a later switch.
- **Rhythm/meter:** probabilistic metrical tree; metrical weight = tree rank; held/whole/multi-bar structural.
  Tree sampled **per repetition-class** (restatements inherit the rhythm; §15 Fix 3), after the form.
- **Harmonic rhythm:** **adaptive by the tree** (sampled rate; chord changes at nodes of the matching depth).
- **Motif:** **render-once-then-vary** (seed-selected) + operator variation; stored as anchored figure-n-gram
  sequences; abstract schema deferred.
- **Variation similarity:** operator-orbit generate + edit-distance retrieval; perceptual/embedding metric later.
- **Surface:** keep FigureNGram + I-projection + TV objective + constraint engine.
- **Skeleton binding:** hard alignment of harmony to the metrical/time-span tree.
- **Two hands:** share Form/Harmony/skeleton; separate surface + motif; re-coupled by `hand_coupling`.
- **Controls:** global per-piece values (no intra-piece variation in v1); each dial parameterizes one layer (§12).

## 10. Data-derived vs hand-specified vs deferred (at a glance)
- **Learned from corpus (cheap/safe):** form repetition-structure prior; opening/closing positional stats;
  harmony rule probabilities (inside-outside); figure surface (already); base-duration distribution (already).
- **Hand-specified (the idiom / few interpretable params):** the harmony *rule set*; the meter split grammar;
  the variation operator set; the style knobs.
- **Deferred (hard/fragile):** unsupervised harmony rule-set induction; adaptor/fragment-grammar unit induction;
  abstract motif-schema learning; named-template form; modulation; phrase-boundary detection as ground truth.

## 11. Deep dive 1 — the repetition-structure prior (the coherence engine)

*Developed first because it reshapes the architecture the most: it adds the missing apex of the hierarchy
(the current model has no form concept), it is the mechanism of internal coherence (the #1 gap), and it defines
the segments + parallelism links the harmony and motif layers consume. Seams to those layers are **OPEN**, not
locked.*

### 11.1 One hierarchy, two scales
There is a single hierarchical time structure per piece (GTTM: grouping ⊕ metrical ⊕ time-span, unified). Its
**coarse levels are form** (segments, optionally grouped, with parallelism links); its **fine levels are the
metrical subdivision** (Layer 2); **harmony attaches at an intermediate harmonic-rhythm level**. "Form" is not a
separate timeline glued on — it is the *top of the same skeleton*. This layer designs the coarse levels + the
**repetition relation**, the genuinely-new coherence-bearing object.

### 11.2 The object we sample: a canonical repetition string
A piece of `B` bars at segment granularity `g` has `n = B/g` base segments `s₁…sₙ`. The repetition structure is a
**labeling** in which same-material segments share a base letter and variants take primes — e.g. `A A B A`,
`A B A B`, `A A′ B A″`. The form layer **samples this string** from a corpus-estimated distribution, then marks
opening/closing bias. The string yields the `restates` links (segment → earlier class it restates) and a per-link
`SAME`/`VARIANT` flag feeding the motif `variation_budget`.

### 11.3 Estimation (offline): corpus → `P(structure)`
Per corpus piece, for each candidate `g`:
1. **Segment** on the metric grid (`g` bars/segment).
2. **Similarity** `sim(sᵢ,sⱼ)∈[0,1]` from the existing figure n-grams — Jaccard on each segment's figure multiset,
   or normalized edit distance between the segments' figure-sequences, combined across hands; computed in
   **anchor-relative space modulo a global transposition** so a transposed restatement counts as the same idea
   (this invariance is itself a modeling choice — §11.6).
3. **Quantize** by thresholds: `sim ≥ τ_same → SAME`; `τ_var ≤ sim < τ_same → VARIANT`; else `DIFFERENT` (set
   `τ` from the corpus similarity histogram, ideally at its bimodal valley).
4. **Canonicalize** greedily L→R: segment `i` takes the label of the earliest `j` with `sim(i,j) ≥ τ_var` (prime
   if `VARIANT`); else a new letter.
5. Record `(n, g, string)`.

**Aggregate** into `P(string | n, g)` and `P(g | n)`. **Back-off** for sparse cells (small corpus): string →
"novel / repeat-of-k-back per segment" sequence → first-order segment-label Markov → `(#distinct-classes,
repeat-rate)`. This chain prevents collapse to trivial all-same / all-distinct structures. Separately count the
decoded harmonic function / figure family at the **first** and **last** segment → `P(open)`, `P(close)` (a soft
bias, not an enforced cadence).

### 11.4 Sampling (generation)
Given `B`: draw `(g,n) ~ P(g|n)` consistent with `B`; draw `string ~ P(string|n,g)` with back-off; emit
`SegmentNode`s with class labels, `restates` links + `SAME`/`VARIANT` flags, and opening/closing bias on `s₁`/`sₙ`.
**No named form is asserted** — the macro-shape is whatever the corpus's recurrence statistics produce.

### 11.5 Seams to the layers below (OPEN — deliberately not locked)
- **Fresh class** → its own metrical tree + harmony + a freshly-rendered motif seed.
- **Repeat/variant** → reuses the class's metrical tree + motif under variation; **its harmony may differ** (the
  period case: antecedent→HC, consequent→PAC over the *same* idea). Whether a restatement copies, transposes, or
  re-harmonizes is an **interaction to resolve in deep dive 2 (harmony/skeleton)** — flagged, not fixed.
- **`SAME` vs `VARIANT`** → operator concentration in deep dive 3 (motif).

### 11.6 Limitations (layer-specific)
- **Threshold sensitivity:** `τ_same/τ_var` decide the structure; mis-set → wrong parallelism. Mitigate via the
  corpus histogram + validation against the §8 structural metrics.
- **Sparsity:** `P(string|n,g)` thins on a small corpus; the back-off risks over-smoothing toward trivial structures.
- **Flat segmentation:** a fixed `g` can't express nested grouping (a 2-bar idea inside a 4-bar phrase);
  multi-level grouping trees are a refinement, not done now.
- **"Same idea" definition** is the transposition/variation-invariance choice in §11.3.2 and inherits the
  edit-distance perceptual-flatness limit (§7 L3).
- **Figure-content only:** learns recurrence of *figures*, not *harmonic* form; the harmonic side is the §11.5 seam.

## 12. Controls (the founding style API) — preserved and re-homed

The redesign must keep the controllability that motivated the project (`docs/generator.md` §1, §7). All survive;
most are *improved* by acting on **structural units** instead of independent grid cells (the fix for "random draws
ignore structure"). Each control is a parameter of **one** layer's generative process, so they stay orthogonal.

| Control | Question | New home (layer) | Note |
|---|---|---|---|
| Hand activity / overlap `h_o` | both hands vs one alone | **region-activity gate** over the skeleton; Gaussian copula couples the hands' per-region gates (`ρ=2h_o−1`) | per region, not per cell |
| Co-activity / sync `h_s` | attacks **& offsets** align or conflict | **sharing of the two hands' surface metrical subtrees** | shared subtree ⇒ coincident on/offsets |
| Metric density | notes per bar | **metrical-tree subdivision depth + SOUND/TIE/REST leaf probs**, per hand | metrically aware |
| Layout / texture | melodic / chordal / bass | **per-hand render mode** over the HarmonyTree | chordal/bass now coherent over a real grammar; can vary per segment |
| Register home/spread | which register | register-curve prior (demoted) | unchanged |
| Harmonic palette | which chords | chord vocabulary feeding the grammar | unchanged/enhanced |
| `commonness_bias` β | common vs rare figures | surface tilt | unchanged |
| `variation_budget` | how much restatements vary | motif operator concentration | NEW (deep dive 3) |

**Orthogonality** holds because each dial parameterizes a distinct layer (region gate / subtree-sharing /
subdivision grammar / render mode / harmony grammar), exactly the separation §7 demanded.

**LOCKED — control granularity: global per-piece.** Each style knob takes a single value for the whole piece
(matches the original API). Controls still act on structural units (region gate, subtree-sharing, subdivision
grammar, render mode), but their *values* are constant across the piece; intra-piece variation (density
acceleration, per-phrase texture) is **not** modeled in v1.

## 13. Deep dive 2 — the metrical/time-span skeleton and the harmony grammar

*Developed second: it establishes the coordinate substrate (the skeleton) that form/motif reference, unifies
harmony+rhythm on one structure, and resolves deep dive 1's open seam (how a restatement re-harmonizes).*

### 13.1 The metrical/time-span skeleton (precise)
Per segment, a recursive **meter-aligned subdivision tree** `G_meter`:
- Root = segment span; multi-bar nodes split into bars / 2-bar groups; a bar node splits by **meter-determined**
  groupings (4/4→`{½,½}`; 3/4→`{¼,¼,¼ | ¼,½ | ½,¼}`; 6/8→`{⅜,⅜ | ¼,¼,¼}`); sub-bar nodes split binary/ternary by
  the remaining duration. Each split option carries a probability (fittable).
- A node **terminates** (→ leaf) with probability rising as it gets shorter/weaker (density knob); leaf type ∈
  `{SOUND, TIE, REST}` by a categorical on metrical weight + density.
- **Metrical weight = GTTM tree rank** (root strongest; first sibling stronger; deeper = weaker): a total preorder
  over onsets, replacing `gcd(k,M)/M` (agrees on regular meters; handles compound/asymmetric).

### 13.2 The harmonic-rhythm frontier (adaptive)
The **harmony slots** = a **frontier** (an antichain covering the span): descend from the root, stop at each node
whose span ≤ `1/harmonic_rate` (sampled). This yields `m` ordered slots, possibly *irregular* (a ½-bar chord then
two ¼-bar chords — harmonic rhythm follows the rhythm tree). Because the frontier is a cut through the *same* tree,
every chord sits on a metrical boundary by construction — the "no `if`s" harmony↔meter coupling.

### 13.3 The harmony grammar: a projective tree over the `m` slots
The **`HarmonyTree` is a projective tree whose `m` leaves ARE the `m` harmonic slots in temporal order**; internal
nodes carry functional categories, leaves carry surface `Chord`s. Generation = top-down sampling of a tree with
exactly `m` leaves (standard length-`m` PCFG generation; trivial for `m ≤ 16`). Compact single-key GSM:
- Head `Seg → T`; prolongation `X → X X` (`X∈{T,D,S}`); preparation (left-branching) `T → D T`, `D → S D`;
  instantiation `T→t`, `D→d`, `S→s`; substitution (pruned, → key-relative `Chord`) `t→{I,vi}`, `d→{V,V7,viiº}`,
  `s→{IV,ii,ii6}`.
- Production probabilities `θ` learned (§13.5). Modulation (`X_key→T_key′`) deferred behind a switch.

### 13.4 Cadence & opening (closure as a feature, not an `if`)
The form supplies each phrase a **closing pattern** (§19) — a functional suffix that **constrains the rightmost
derivation path** (e.g. `(D, T)` authentic, `(S, D)` / `(D,)` half, `(S, T)` plagal); the closing chords come
from the realization distribution, so deceptive (closing tonic realized as `vi`) emerges rather than being a named
type. The first phrase opens on the tonic; the terminal phrase closes to the tonic. All carried as node features
from the form — never `if last_bar`.

### 13.5 Fitting `θ` (inside-outside + Dirichlet)
Decode each corpus piece's chords (the **kept** Viterbi decoder), map scale-degree chords → functions
(`I,vi→t`; `V,viiº→d`; `IV,ii→s`), and run **inside-outside / EM** over these sequences on the **fixed** rule set,
with **Dirichlet smoothing** and **theory-initialised** `θ` (local-optimum mitigation); prune rules that never
fire. (Inducing the rule *set* is the fragile path we avoid — §7 L1.)

### 13.6 Resolving deep dive 1's seam — how a restatement re-harmonizes
The form's `restates` link binds the **motif/figure content** (the surface); **harmony is derived per segment**
under its own `cadence_target`. A consequent that restates the antecedent's idea ends on `PAC` while the
antecedent ended on `HC` — the classical period **emerges because the cadence feature differs**, not by a coded
rule, while the figures are the same idea re-grounded. Default: independent per-segment derivation (only the
cadence diverges, so most slots match). **OPEN/tunable:** optionally enforce **harmonic-prefix sharing** between
restated segments for tighter parallelism — flagged, not locked.

### 13.7 Limitations (layer-specific)
- **Functional reduction to T/D/S is coarse** (blurs IV vs ii, secondary function); substitution recovers some
  color but the single-key/triadic ceiling holds.
- **`θ` is a corpus+prior blend:** small corpus → sparse counts → Dirichlet/theory priors carry weight; "learned"
  is partly prior-driven (honest).
- **Hard frontier alignment** forbids harmony syncopated against the meter (chords change only on metrical nodes)
  — usually desirable here, but a limit.
- **Decoder noise** in the corpus chords propagates into `θ`.
- **Small `m`** (very slow harmonic rhythm) limits derivation depth → limited functional structure.
- **Projective/context-free:** no crossing harmonic dependencies, no context-sensitive voice-leading (§7 L1).

## 14. Deep dive 3 — motif variation (the surface-coherence bridge)

*Developed third: bridges the plan (form repetition classes + skeleton + harmony) to the preserved figure
surface, turning "segment B restates A" into concrete, recognizable-but-not-identical material.*

### 14.1 What a motif is (two representations)
- **`MotifSchema` (reusable identity, fully relative):** an ordered sequence of relative `FigureNGram`s + the
  inter-figure relative anchor offsets (contour across figures) + the relative rhythmic structure. Anchor- and
  time-relative → transposes/retimes for free. This is "a sequence of n-grams" abstracted from its grounding.
- **`MotifInstance` = `MotifSchema` + grounding:** absolute anchor of the first figure, the metrical leaves it
  occupies, and the chords at those leaves (the "anchored, time- and chord-grounded" part).
Reuse = strip grounding, transform the schema, re-ground.

### 14.2 Seed creation + seed-selection (mitigating render-once)
A repetition-class's **first** occurrence is rendered concretely via the surface (Layer 5): per metrical slot,
draw a figure by the I-projection tilt given (register-prior anchor, HarmonyTree chord, metrical weight). Draw
`N_seed` candidate `MotifInstance`s, keep the best (or sample ∝ quality) by a **quality score** `Q` = weighted
sum of: mean metrically-weighted `harmonic_fit`; contour smoothness (penalize large inter-figure register jumps);
rhythmic fit to the metrical tree; figure typicality. The extracted `MotifSchema` = `prototype_A`. *(Seed-selection
chooses among i.i.d. renders of one segment — not autoregressive conditioning, so it doesn't reintroduce
aimlessness.)*

### 14.3 Variation operators (the orbit), in schema space
Each maps `MotifSchema → MotifSchema`: `IDENTITY`; `DIATONIC_TRANSPOSE(Δ)` ("same rhythm, new register");
`SEQUENCE(Δ,k)` (k transposed repeats — sequence/continuation); `INVERT` (negate relative steps); `RETROGRADE`
(reverse, low prob); `AUGMENT(c)/DIMINISH(c)` (scale durations — new time resolution, longer/shorter slots);
`ORNAMENT(type)` (insert passing/neighbor figures from the vocabulary — raises `n` & density); `FRAGMENT(sub)`
(contiguous sub-sequence — fragmentation); `TRUNCATE/EXTEND` (fit a different slot count). Operators **compose**
(orbit = the monoid they generate). The deep-dive-1 `SAME`/`VARIANT` flag + `variation_budget` set the operator
distribution: `SAME → {IDENTITY, DIATONIC_TRANSPOSE}`; `VARIANT →` a sampled composition, concentration by budget.

### 14.4 Re-grounding to a new segment (anchor / time / chord)
Reusing a schema in segment `B`: (1) **place** the (operator-altered) rhythm onto `B`'s metrical slots,
`base_duration` from the slot spans (`AUGMENT`/`FRAGMENT` change which/how many slots; a `SAME` repeat reuses
`A`'s tree); (2) **anchor** from `B`'s register prior (+ any `DIATONIC_TRANSPOSE`); (3) **re-harmonize** — `B`'s
slots carry `B`'s chords (deep dive 2; may differ only at the cadence), so the schema's figures keep their relative
contour but are **re-scored against `B`'s chords** via §14.5 and adapt harmonically at the cadence automatically
("the same idea in a new harmonic context").

### 14.5 `similarity_fit` + edit-distance neighborhood (coherence ↔ reference)
The operator output is an **intended** figure per slot; we don't emit it verbatim. We draw from the empirical
vocabulary by the tilt with a new term
`p(f) ∝ c(f)^β · exp(λ_curve·slope_fit + λ_harmonic·harmonic_fit + λ_accent·accent_fit + λ_similarity·similarity_fit)`,
`similarity_fit(f) = −edit_distance(f, intended)` over `N_ε(intended)` (vocabulary figures within edit-distance ε
in `(degrees, normalized_duration)` space). This (i) keeps the figure **recognizably close to the motif**
(coherence) and (ii) keeps it a **real corpus figure** (TV-calibration intact). `λ_similarity` is the
coherence↔reference dial: `0 →` corpus marginal; `→∞ →` the intended figure. If the operator output isn't in the
vocabulary, retrieval **snaps** to the nearest real figures.

### 14.6 Limitations (layer-specific)
- **Render-once propagation:** the seed anchors a class's quality (mitigated, not removed, by seed-selection).
- **Orbit coverage = operator-set design** (§7 L3); some natural variations unreachable.
- **Edit-distance perceptual flatness** (§7 L3): `N_ε` may include perceptually-distant equal-cost figures.
- **`ORNAMENT`/`AUGMENT` change `n`/slot occupancy** → must re-fit `B`'s metrical tree; on failure, fall back
  (fewer ornaments → identity).
- **`similarity_fit` vs `harmonic_fit` tension** at a cadence where `B`'s chord differs: staying near the motif
  fights fitting the new chord; the `λ`s arbitrate, yielding a compromise.
- **Within-class only:** coherence is `A, A′, A″` (within a class); **cross-class thematic derivation** (a
  contrasting idea B *derived from* A) is **not** modeled — a deferred refinement. Real music has both.

## 15. Coherence review — one 8-bar piece through every layer

*Tracing a worked example end-to-end to verify the layers compose. Found three genuine cross-layer issues (all
fixable, all data-driven, none adding stylistic bias); the rest composes.*

**Setup.** C major, 4/4, 8 bars; global controls: hands overlap (`h_o` high), `h_s` medium, RH = MELODIC, LH =
SUSTAINED_BASS, harmonic_rate = 1 chord/bar, `variation_budget` tight.

**Trace.**
- **Form (§11):** `g=2` bars → `n=4` segments; draw string `A A′ B A″` → `restates`: s2→s1 (VARIANT), s4→s1
  (VARIANT), s3 fresh (B); opening bias on s1, closing PAC on s4.
- **Skeleton (§13):** one metrical tree **per class** (Fix 3) — `tree_A`, `tree_B`; harmonic frontier at 1/bar →
  2 slots per 2-bar segment → 8 chord slots.
- **Harmony (§13):** rooted **per phrase** (Fix 1) — phrase1 = s1+s2 (HC), phrase2 = s3+s4 (PAC):
  e.g. `I I | IV V ‖ I vi | V7→I`. The classic period emerges.
- **Motif (§14):** seed `prototype_A` from s1 (best of N renders), `prototype_B` from s3; s2 = A re-grounded onto
  IV–V (cadential), s4 = A re-grounded onto the PAC; figures adapt via `similarity_fit` + `harmonic_fit`.
- **Hands:** RH renders the motifs; LH (SUSTAINED_BASS) realizes the HarmonyTree roots held per slot; shared
  harmony ⇒ vertically coherent; `h_s` governs RH-figure vs LH-bass attack alignment.
- **Encode + gate:** EventStream → tokens → `GenerationConstraintState`; resample within slot on rejection.

**Findings (cross-layer issues caught):**
1. **Flat segmentation under-determines the phrase structure harmony needs.** Harmony must root and cadence at the
   **phrase**, but flat segments don't say which boundaries are phrase ends. **FIX:** promote the form to a
   **minimal 2-level grouping** — segments group into phrases; **parallelism (`restates`) stays at the segment
   level; harmony-rooting + cadence at the phrase level.** Phrase-length and per-position cadence distributions are
   **learned from the corpus** (still low-bias, no named templates). The "nesting" deferred in §11.6 is therefore
   **not fully optional** — ≥2 levels are required for harmony to be a phrase-spanning arc rather than a per-2-bar
   tonic reset.
2. **Cadence targets were set only for first/last segment.** Middle phrases need them too. **FIX:** extend §11.3's
   positional statistics to a full **`P(cadence_target | phrase position, count)`** learned from the corpus.
3. **Metrical tree was per-segment, but motif restatement must reuse rhythm.** A restatement that resamples its own
   tree wouldn't preserve "the same idea (usually same time resolution)." **FIX:** sample the metrical tree **per
   repetition-class**, not per segment; **restating segments inherit their class's tree**, varied only by explicit
   operators (`AUGMENT/DIMINISH/ORNAMENT/FRAGMENT`). Reconciles §13 with §14.1 (where "relative rhythmic structure"
   is already part of `MotifSchema`) and fixes ordering: **trees sampled per class, after the form.**

**Clarification (made explicit):** harmonic **slots** come from each segment's metrical **frontier**; the harmony
**grammar** roots per **phrase**, spanning that phrase's concatenated slots.

**Confirmed coherent:** restatement re-harmonization (§13.6/§14.4); `similarity_fit` ↔ TV (`λ_similarity=0` ⇒
corpus marginal, §14.5); two hands (shared form/harmony, separate motif; texture hand follows harmony, §4);
seed-selection is non-autoregressive (§14.2); constraint gate with within-slot fallback (§4; caveat L4 — a bad
restatement anchor can only locally degrade, not re-plan).

**Net:** the design **composes after three modest, data-driven fixes**, none adding stylistic bias. Sound
end-to-end.

## 16. Concrete resolution of the §15 findings

### 16.1 Fix 1 + 2 — inducing phrases & cadences from the corpus (no named templates)
A **phrase = a maximal run of segments ending at a cadence**, so phrase boundaries *are* cadence locations —
induce both from data:
- **Offline:** decode each corpus piece's harmony (the **kept** Viterbi decoder); detect **cadence events** —
  authentic (`D→T`) and half (arrival/dwell on `D`) at metrically strong nodes, corroborated by a rhythmic stop
  (long note) and by the repetition string (a returning/new segment often starts a phrase). Record (a) the
  **phrase-length distribution** (bars & segments between cadences) and (b) **`P(cadence_type | phrase index,
  n_phrases)`**, `cadence_type ∈ {PAC, IAC, HC, DC}`.
- **Generation:** after sampling segments + repetition string (§11), **partition the segments into phrases** by
  the learned phrase-length distribution (last phrase ends at the final bar); draw each phrase's
  `cadence_target ~ P(cadence_type | position)` (final → PAC); root the harmony grammar (§13) per phrase.
- **Robustness:** when cadence evidence is weak (short/noisy pieces) fall back to a prior favouring phrase lengths
  that divide `bar_count` (2/4/8) and to `P(cadence_type | is_final)`.
- **Limits:** cadence detection inherits decoder noise; half-cadence ambiguity; sparse `(i, n_phrases)` cells need
  back-off; the grouping is single-level (phrase ⊃ segment) — deeper nesting (period-of-periods) still deferred.

### 16.2 Fix 3 — per-class metrical trees + the canonical generative order
Not an estimator — a **scoping + ordering** fix. The metrical tree is a property of a repetition-**class**: sampled
once per fresh class, inherited by its instances; a VARIANT instance applies rhythmic operators to the class tree;
the seed is rendered onto it. This pins the **canonical generative order** (no forward dependency — each step
consumes only frozen earlier outputs, so the top-down / no-autoregression principle holds):

1. **Form** — sample `(g,n)`, repetition string → segments + classes + `restates`/`SAME`/`VARIANT`; partition into
   phrases; assign per-phrase `cadence_target` (§11, §16.1).
2. **Per-class metrical trees** — one `G_meter` tree per fresh class (§13.1); each segment's harmonic **frontier**
   → its chord slots.
3. **Per-phrase harmony** — GSM tree over each phrase's concatenated slots under its `cadence_target` (§13.3–4) →
   a chord per slot.
4. **Per-class motif seed** — render the class's first instance onto its tree + harmony; seed-select best of `N`
   (§14.2) → `MotifSchema`.
5. **Per-instance render** — restating instances reuse the schema under variation operators, re-grounded to the
   instance's anchor/slots/chords (§14.3–4); draw concrete figures by the I-projection + `similarity_fit` (§14.5).
6. **Encode + gate** — EventStream → tokens → `GenerationConstraintState`; within-slot resample on rejection.
7. **Two hands** — steps 2,4,5 per hand (shared form/harmony from 1,3); re-couple by `hand_coupling`.

## 17. Phased implementation plan (each phase independently testable)

Dependency-respecting and mostly additive; the existing generator path stays until the new path validates.
Conventions (`docs/guidelines.md`): frozen pydantic for serialized objects, dataclasses for internal state,
verbose names, `Final` constants, `uv run` tooling, tests mirror the package layout; `synthetic` may depend on
`n_grams`/`decoder`, never the reverse.

- **Phase 0 — Entity model + metrical/time-span skeleton (no corpus).** `MetricalTree` + `G_meter` sampler
  (hand-set split probs), metrical-weight-by-rank, harmonic-rhythm frontier; entity stubs
  (`FormTree/SegmentNode`, `HarmonyTree`, `MotifSchema/Instance`, `EventStream`). *Verify:* valid subdivisions;
  frontier antichain covers the span; SOUND/TIE/REST + whole-note + cross-bar TIE expressible; weight = `gcd(k,M)/M`
  on regular meters.
- **Phase 1 — Harmony grammar (hand-set `θ`).** Compact single-key GSM as a length-`m` projective-tree sampler with
  `cadence_target` constraints; leaves → key-relative `Chord` (reuse `harmony/expansion`). New path replacing
  `processes/chord_track.py`. *Verify:* well-formed derivations; PAC/HC realized; the period `I I IV V | I vi V7 I`
  reachable.
- **Phase 2 — Top-down surface renderer (preserved I-projection).** Replace the greedy `substitution/generator.py`
  walk with a renderer over a metrical tree + harmony: figures via the tilt (extend `scoring.py` to read the slot's
  chord), `base_duration` from the leaf span, TIE/REST leaves direct, gate via `GenerationConstraintState`. No
  form/motif yet. *Verify:* valid Segments; held/whole notes appear; **figure TV at `λ_sim=0` matches corpus**;
  points 3/5/6 fixed even pre-form.
- **Phase 3 — Form (learned) + phrase/cadence induction.** Offline estimators (`synthetic/fitting/`):
  repetition-string histogram; cadence detection → phrase-length + `P(cadence_type|pos)` (§16.1). Generation:
  sample `FormTree`; wire `cadence_target` into Phase 1. *Verify:* sampled forms match corpus repetition + cadence
  statistics; the 8-bar period reproduces.
- **Phase 4 — Motif reuse-with-variation.** Per-class trees (§16.2 order); seed render + seed-selection `Q`;
  variation operators; re-grounding; `similarity_fit` + edit-distance neighborhood. *Verify:* restatements
  recognizable but not identical; repeated-figure-family & variation-after-repeat rates match corpus; **TV
  preserved at `λ_sim=0`**.
- **Phase 5 — Couplings, controls, calibration.** Re-home `hand_coupling` (region gate + subtree-sharing); wire the
  global knobs; fit harmony `θ` (inside-outside + Dirichlet); calibrate `λ`s (incl. `λ_sim`) vs TV + structural
  metrics. *Verify:* controls move output as intended; TV under target; structural suite logged.
- **Phase 6 — Fit, validate, document.** Fit all learned distributions; full structural + TV metrics; human
  inspection (notation / piano-roll / audio); finalize **`docs/generator-coherence.md`** (as-built) and update
  `docs/generator.md` / `generator-model.md`.
- **Retirement:** once Phases 2–5 validate, remove the old greedy path, `processes/chord_track.py`'s sampler, and
  the accent field's structural role (it survives only as the density prior).

## 18. As-built notes (implementation)

### Metrical / time-span skeleton — `synthetic/structure/meter.py`
Bar groupings are **derived** from `metrical_factors(num, den)` — a simple/compound classification plus the prime
factorisation of the beat count (`prime_factors`, `musak_shared/misc.py`) — superseding §13.1's duration-based
binary/ternary sketch. Node weight matches `gcd(k, M) / M` on simple meters and follows the subdivision on
compound meters. `MetricalGrammarConfig` loads from `configs/generation/metrical_grammar.yml`.

**Deferred — revisit:**
- **Tuplets** are not representable: a node splits binary, or compound-ternary derived from the meter. Adding
  irregular tuplets (eighth-triplets in a simple meter, quintuplets, …) needs an alternative branching option per
  node plus non-dyadic durations (1/12, 1/20, …) in the duration vocabulary (see §7 L2).
- **Asymmetric / odd meters** fall back to a flat prime split (5/4 → five beats, 7/8 → seven); idiomatic groupings
  (5/4 as 2+3, 7/8 as 2+2+3) await corpus evidence. The meter is consulted at the bar level by necessity — bar
  duration alone cannot separate 6/8 from 3/4 (both span 3/4).

## 19. Cadence — a measured closing pattern (data → stochastic model)

*Supersedes the `CadenceType` enum and the `PAC/IAC/HC/DC` labels referenced in §9 / §13.4 / §16.1: those names
are descriptive, not generative primitives.*

**Cadence is a measured pattern, not a category.** A cadence is the **harmonic closing pattern observed at a
phrase boundary** — a short *functional suffix* landing on a metrically strong, rhythmically articulated beat:
`ClosingPattern = (functional suffix f(t*−k … t*), metrical level of t*)`. "PAC / half / deceptive / plagal" are
just frequent suffixes (`…D→T`, `…→D`, `…D→` tonic-realized-as-`vi`, `…S→T`) — descriptions of patterns we count,
never an enumerated primitive. This mirrors the form layer (count repetition strings, don't name AABA).

**Perfect vs imperfect is excluded for now** — it is a *voicing* axis (root-position bass, tonic in soprano), and
the Viterbi chord track is decoded from pitch-class content with no register. It is recoverable later from the raw
notes via a bass/soprano pass, as an additive `strength` field; it is not a generative primitive.

**Retrieval from data (non-circular).** Detecting cadences *by* assuming `D→T`, then "discovering" cadences are
`D→T`, is circular. Separate the cue from the payload:
- **Boundary cue (non-harmonic):** metrical strength (metrical-tree weight at `t*`), rhythmic articulation (a long
  note / rest / large IOI at `t*`), and repetition structure (a returning/new segment tends to start a phrase).
- **Harmonic-arrival cue (general relaxation, not a `D→T` rule):** tonal tension *drops* onto a stable chord at
  `t*`. Tension is gradient — **tonic-triad pitch-class overlap** (locked choice) — so it fires for `D→T`, `S→T`
  (plagal) and others without presupposing the progression.
- A **cadence** = a metrically-strong, rhythmically-articulated boundary co-located with harmonic relaxation onto
  a stable chord. High *precision*, modest recall (only confident cadences are needed to estimate distributions).
- At each detected cadence, **record the payload** — the closing functional suffix and the metrical level. The
  harmonic content of cadences is thus *measured*, not assumed (plagal closes are discovered if present).

**The stochastic model.** Counting detected cadences yields exactly two distributions:
- `P(phrase_length)` — bar distances between cadences;
- `P(ClosingPattern | phrase_position)` — antecedents close open (`…→D`), final phrases close to tonic (`…D→T`).
At generation: sample a phrase partition, then per-phrase a closing pattern; the harmony grammar (§13) forces the
rightmost leaves to the suffix's functions, and the closing **chords come from the realization distribution** — so
*deceptive* (closing tonic realized as `vi`) emerges rather than being a named type.

**Merge with the neural model (the shared skeleton).** `ClosingPattern` — with the metrical tree, the
harmonic-function track, and the phrase boundaries — is the CAST-style **skeleton**. Annotate the corpus once with
this skeleton; the **stochastic** generator then turns the annotations into *counted distributions* and renders
texture via the grammar + figure surface, while the **neural** model *conditions on / trains against* the same
skeleton labels. They are two interchangeable texture-renderers over one shared skeleton vocabulary — the merge is
"share the skeleton annotation, swap the renderer," and `ClosingPattern` is part of that contract. (This is why
its measurable, learnable shape matters now, before data structures harden.)

**Locked decisions.** (1) `ClosingPattern` = a functional suffix of **terminal + one approach** function (`(D, T)`
authentic, `(S, T)` plagal, `(S, D)` / `(D,)` half), length extensible. (2) **Purely functional** — the closing
chord is drawn from the realization distribution, so authentic vs deceptive differ only by the realized tonic
chord, not by type. (3) Harmonic-arrival tension = **tonic-triad pitch-class overlap**. (4) **Voicing /
perfect-imperfect deferred** to a later bass/soprano pass.

## 20. Surface render — figures on the metrical tree (Phase 2)

*Elaborates §4 Layer 5: how the preserved figure surface meets the metrical/time-span skeleton.*

**Slot model.** A metrical-tree **leaf is a render slot.** A `SOUND` leaf hosts **one figure of any `n ≥ 1`** (an
`n=1` figure *is* a single note — no special case), scaled to the leaf's span
(`base_duration = leaf_span / figure_normalized_span`) with every onset duration `≥ shortest_note_duration`. A
`TIE` leaf **extends** the previous note; a `REST` leaf is silence. The **harmonic frontier** (§13.2) gives each
slot its chord; the concrete figure is drawn by the I-projection tilt reading that chord and the leaf's metrical
weight.

**Rhythm is two complementary levels:** the tree places figures, rests, and ties in metrical time (macro); each
figure supplies its within-slot notes (micro). `base_duration` comes from the *slot span*, never fit-to-remaining-
bar — removing the old compression-to-short-notes bias (points 3, 5).

**Single / whole notes.** `n=1` is just a single note; a whole-note-per-bar = a bar-sized `SOUND` slot holding an
`n=1` figure. This requires `min_n=1` in figure extraction (`configs/analysis/n_grams.yml`) — set. The signature
builder already handles a single-onset window (normalized duration `(1,1)`, `base_duration` = the note's held
duration).

**Cross-bar (concrete — not "hypermeter").** The barline is always a metrical-tree boundary. (a) **Sustained /
ligature crossings** of any offset and span → `TIE` leaves: a `SOUND` attack before the barline plus `TIE`
continuation after. (b) **Multi-onset gestures across bars** (e.g. a 7-note idea over 2 bars) → a **motif**
(Phase 4): a metrical subtree spanning the bars plus its contour, reused as a unit. A single beamed multi-onset
*figure* deliberately stays within a bar (matching notation: beams don't cross barlines, ties and slurs do).

**Two duration bounds.** `min_leaf_duration` (the metrical grammar's minimum *slot*) and `shortest_note_duration`
(the minimum of the duration vocabulary), with `min_leaf_duration ≥ shortest_note_duration`. The renderer bounds a
slot's figure so every onset is `≥ shortest_note_duration`; a slot too small for an `n ≥ 2` figure takes an `n=1`
note — which only works because `n=1` is admitted.

**As-built (Phase 2, `synthetic/render/`).** `render_slots` pairs each metrical-tree leaf with its
harmonic-frontier chord; `SurfaceRenderer` walks the slots per bar and hand, emitting figures (`SOUND`, scaled to
the slot span, drawn by the I-projection tilt over the vocabulary entries that *fit* the slot), holds (`TIE`), and
rests (`REST`), each gated by `GenerationConstraintState` with a rest fallback. `base_duration = slot span ÷
figure span-units` (no greedy fit-to-remaining-bar). Phase-2 simplifications, superseded later: both hands share
one metrical tree and chord track (per-hand trees, the shared-skeleton/surface split, and hand couplings are
Phase 5); the tilt's harmonic term is **duration-weighted** (`figure_selection._chord_tone_coverage`) — the
metrical-position refinement of `scoring.harmonic_fit` is deferred; `λ_similarity` / motifs arrive in Phase 4. At
`λ_curve = λ_harmonic = λ_accent = 0` figure choice is count-proportional, preserving the corpus marginal.

## Sources
Form/segmentation & repetition: Cambouropoulos LBDM (ICMC 2001); Pearce/Müllensiefen/Wiggins segmentation
comparison (ISMIR 2008) & IDyOM (2012); Meredith SIA/SIATEC/COSIATEC; self-similarity-matrix MSA; Sidorov/Jones/
Marshall "Music Analysis as a Smallest Grammar Problem" (ISMIR 2014). Harmony grammar & induction: Rohrmeier 2011
(*J. Math & Music* 5(1)) & Rohrmeier–Moss (extended tonal harmony); Tsushima et al. "self-emergent grammar of
chord sequences" (arXiv 1708.02255); "Unsupervised Induction of Harmonic Syntax for Jazz" (TISMIR 2025);
inside-outside / PCFG-EM (standard). Probabilistic grammars for music: Gilbert & Conklin (PCFG melodic reduction);
Nakamura/Hamanaka probabilistic GTTM. Grammar induction theory: Johnson/Griffiths/Goldwater adaptor grammars;
O'Donnell fragment grammars; Harasim et al. Jazz Harmony Treebank (ISMIR 2020). Rhythm trees: Rohrmeier ISMIR
2020; Foscarin/Jacquemard/Rigaux. Structure/meter: Lerdahl & Jackendoff 1983 (GTTM). Variation/recombinance: Cope
(EMI, recombinant music); Müllensiefen & Frieler (melodic similarity); Rizo & Iñesta (tree grammars / edit
distance); survey arXiv 2403.07995 (structure in symbolic generation).
