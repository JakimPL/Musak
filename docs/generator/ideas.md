# Synthetic Exercise Generator: Design

This document describes a classical, stochastic, data-driven generator of sight-reading piano exercises. Its purpose is
to produce synthetic data that complements the LLM-based `musak_model`, while remaining controllable, interpretable, and
faithful to the figure statistics of the training reference.

The generator is intentionally *not* a neural model. It is built from a small set of **low-order global stochastic
processes** — a register trajectory, a rhythmic-activity field, and a harmonic chord track — feeding a single
figure-substitution step, with the empirical figure vocabulary carrying all local micro-structure. The design exploits
three facts the existing codebase already establishes.

## What the data model already determines

1. **Figures are purely relative.** `build_figure_ngram` (`musak_model/n_grams/figure/builder.py`) anchors pitch to the
   minimum diatonic position of onset 0 and normalizes durations to the shortest onset (`_normalize_durations`). A
   `FigureNGram` is therefore a *translation-invariant contour + scale-invariant rhythm template* — it encodes **micro**
   structure (local shape and relative rhythm) and nothing about absolute register, tempo, metrical position, or
   harmony.

2. **Pitch lives on a diatonic integer lattice**, not semitones:
   `note_diatonic_position = octave_offset · scale_size + (degree − 1)` (`musak_model/n_grams/figure/pitch.py`). The
   global pitch process should live on this 1-D lattice; semitones only re-enter at the constraint layer.

3. **"Similar to reference" already has a precise meaning.** `figure_distribution_metrics`
   (`musak_model/n_grams/profile/metrics/distribution.py`) is the mean **total-variation distance** per
   `(scale_type, hand, n)` group; `figure_profile_comparison_metrics`
   (`musak_model/n_grams/profile/metrics/profile_comparison.py`) adds relative-error on group totals and on
   monophonic / chords-only / in-scale rates. That metric is the generator's objective function.

The key consequence: **the global processes only need to be low-order.** The figures carry the high-order
contour/rhythm structure empirically — and `FigureVocabulary.sample` with `commonness_bias`
(`musak_model/synthetic/figures.py`) already samples them. So the global model only supplies register trajectory,
rhythmic activity, and harmonic context. This is why the approach beats a flat Markov chain over absolute notes (see
[Rejected alternatives](#rejected-alternatives)).

## 1. The reference database

The figure database must store more than relative figures, because the generator needs **conditional** distributions the
current schema cannot express:

- $p(\text{figure} \mid \text{starting degree})$ — required for harmony (Section 4).
- $p(\text{figure} \mid \text{metrical position})$ — required for the accent/placement model (Section 3).
- $p(\text{absolute base duration})$ — required to map a figure's normalized rhythm back to real durations.
- $p(\text{figure} \mid \text{time signature})$.

**Current state** (`musak_model/n_grams/profile/io.py`): the figure DB is a CSV keyed by `(scale_type, hand, n)` →
`count` of a *relative* `FigureNGram`. The `FigureNGram` is already a **lossy projection** — it has discarded absolute
degree, absolute duration, time signature, and metrical position. The rhythm table keeps `time_signature` but is
disjoint from figures.

**Proposed change — store rich, drop on demand.** Refactor `analyze-n-grams` around a **fact table** (one row per
figure occurrence), not aggregated counts:

```
scale_type, hand, n,
figure              # the relative FigureNGram (canonical vocabulary, unchanged)
anchor_degree       # absolute scale degree 1..7 of the anchor
anchor_accidental
anchor_octave       # or full diatonic position
base_duration       # the Fraction divided out by normalization
bar_relative_onset  # metrical position of onset 0 within the bar
time_signature
chord_context       # decoded chord + beat-strength at the figure (see Section 4)
```

`figure/all/counts.parquet` and the existing `FigureProfile` then become **materialized views** — a
`groupby(scale_type, hand, n).count()` projection over the fact table. "We can always drop information" becomes
literally *marginalize over the columns you are not conditioning on*. Parquet handles tens of millions of occurrence
rows comfortably, and aggregation stays cheap. This decouples *what we gather* from *what we count*.

Two deliberate decisions:

1. **Keep the relative `FigureNGram` as the canonical key for the TV-distance metric.** The new columns are strictly
   *additive context*; they must not break `figure_distribution_metrics`, which still operates on the marginal
   projection.

2. **Condition the contour vocabulary and the placement on different axes.** A figure's relative contour is largely
   time-signature- and meter-invariant, so **pool the contour vocabulary across time signatures** (more data per
   figure). But *where* a figure tends to start (`bar_relative_onset`) is strongly meter-dependent, so condition
   **placement** on `time_signature` and `bar_relative_onset`. Conditioning the contour itself on time signature would
   only fragment the counts and raise TV-distance variance for no benefit.

> Caution: storing `anchor_degree` / `base_duration` per occurrence is what makes harmony and rhythm reconstruction
> measurable, but any *aggregated* table built from those axes can suffer cardinality blow-up. Keep the rich table as
> the fact source and aggregate only to the conditioning the generator actually queries.

## 2. The pitch curve

### Unifying frame: a Gaussian process on the diatonic lattice

Model each hand's onset-pitch trajectory $P_i(t)$ as

$$P_i(t) \sim \mathcal{GP}\big(\mu_i,\; k(t,t')\big), \qquad k = k_{\text{arch}} + k_{\text{wobble}},$$

with $\mu_i$ = the hand's home register (the code defines `RIGHT_HAND_HOME_OCTAVE = 5`, `LEFT_HAND_HOME_OCTAVE = 3`),
$k_{\text{arch}}$ a long-lengthscale kernel (the slow phrase arch), and $k_{\text{wobble}}$ a short-lengthscale Matérn-½
kernel (local register drift). Conditioning on a concrete start point is GP posterior conditioning on $P_i(0) = p_0$;
conditioning on octave only means conditioning on a band. Sample continuous, then quantize to the lattice at
substitution time. Every concrete option below is a special case of this GP.

#### Option A — Ornstein–Uhlenbeck / AR(1) (the cheap $k_{\text{wobble}}$)

$$dP_i = \theta(\mu_i - P_i)\,dt + \sigma\,dW \quad\Longleftrightarrow\quad P_{k+1} = P_k + \theta(\mu_i - P_k) + \varepsilon_k.$$

Stationary variance is $\sigma^2 / 2\theta$ (register spread); autocorrelation is $\rho(\tau) = e^{-\theta\tau}$. Fit
$\theta, \sigma$ from two measured numbers: the empirical spread of onset pitch per hand and its lag-1 autocorrelation.
Mean reversion keeps the hand inside playable register instead of drifting out, which a plain GP/random walk will not.
Bounding $\varepsilon_k$ gives the "max melodic gap" softly. Cost $O(N)$, 3 interpretable parameters.

#### Option B — Band-limited random function (the cheap $k_{\text{arch}}$)

$$P_i(t) = m_i(t) + \sum_{j=1}^{J} a_j\,\phi_j(t), \qquad a_j \sim \mathcal{N}\big(0,\, S(f_j)\big),$$

with $\phi_j$ a **DCT/B-spline basis** (not plain sines — this avoids wrap-around at the boundaries) and $S(f_j)$ the
empirical power spectral density of length-normalized training trajectories (Welch-average periodograms after resampling
each exercise to unit length). Small $J$ gives guaranteed smoothness and a *fixed, stable* generator that you fully
control. This is exactly "smooth movement through the entire exercise" = low-pass noise.

#### Recommendation: arch + OU

Use **B around a mean, with A as the residual**: a random low-frequency arch (capturing global shape *and its
variability* from data) plus a mean-reverting, lag-1-matched OU wobble. In GP language this is precisely
$k_{\text{arch}} + k_{\text{Matérn-½}}$; implement it as arch-spline + OU because both are $O(N)$ and each parameter is a
moment read directly off the data.

| pitch model        | matches              | smooth      | stays in register   | discrete-native | params            |
| ------------------ | -------------------- | ----------- | ------------------- | --------------- | ----------------- |
| full GP            | autocovariance       | yes         | needs a cap         | quantize        | kernel hyperparams |
| OU / AR(1)         | spread + lag-1       | rough (C⁰)  | yes (mean-revert)   | easy            | θ, σ, μ           |
| DCT/spline random  | spectrum             | yes (band)  | soft                | quantize        | #bands, gains     |
| **arch + OU (rec.)** | spectrum + lag-1     | yes         | yes                 | quantize        | few               |

The curve's **value** sets the anchor (the absolute diatonic position to drop a figure on); the curve's **local slope**
biases *which* figure (ascending vs descending). That is figure substitution along the curve, made precise — and it
exploits the fact that figures are anchor-relative.

## 3. The accent grid

This is a **marked point process on the duration grid**: each grid cell is onset / no-onset, and onsets carry an accent
weight. The rhythm extractor (`musak_model/n_grams/profile/rhythm/extraction.py`) already measures the strong-vs-weak
onset rate, grid-alignment rate per denominator, onset density, and IOI/duration n-grams — exactly the moments to match.

#### Option A — Inhomogeneous Bernoulli with a metrical intensity

$$\Pr[\text{onset at } t] = \sigma\big(\beta_0 + \beta_1 \cdot \mathrm{ind}(t)\big), \qquad \lambda(t) = \lambda_0 \cdot \mathrm{ind}(t)^{\gamma},$$

where $\mathrm{ind}(t)$ is a **metrical-hierarchy weight** (Barlow indispensability from the meter's prime factorization,
or per-subdivision-level probabilities). $\sum_t \lambda(t)$ is the expected density (the style input $h_d$); the
strong/weak ratio you measure *is* $\lambda_{\text{strong}} / \lambda_{\text{weak}}$, so it is direct moment-matching.
Downside: independent cells ignore sequential rhythm.

#### Option B — Rhythm-figure n-gram chain

Sample a Markov chain over the already-extracted IOI/duration n-grams. Matches the rhythm-n-gram reference by
construction and captures runs (sequential correlation), but produces a discrete token sequence rather than "weighted
impulses," and needs metrical conditioning.

#### Option C — Log-Gaussian Cox process (recommended)

$$\log \lambda_i(t) = \underbrace{\mathrm{ind}(t)}_{\text{meter}} + \underbrace{g_i(t)}_{\text{smooth activity envelope}},$$

where $g_i(t)$ comes from the *same* low-frequency machinery as the pitch curve. This is the "weighted impulses"
picture: onsets are a Bernoulli/thinned draw with probability $\propto \lambda$, and the **accent weight is $\lambda$ at
that cell**, used downstream to bias figure choice (strong → longer/denser figure or chord; weak → passing note). The
smooth $g_i$ produces **phrasing** — auto-correlated busy/sparse regions — which independent Bernoulli cannot. The style
knobs map cleanly: $h_a^i$ = mean of $g_i$, $h_d^i$ = gain on $\lambda$.

Let the **figures carry IOI micro-structure**: you do not need Option B to match rhythm n-grams if substitution respects
figure rhythm. Use Option C for the skeleton and accents, and let figure rhythm fill the detail. The point process also
carries a coarse **activity gate** that the hand co-activity coupling acts on (Section 6).

## 4. Harmony: the latent chord track

Relative figures are harmony-blind, and forcing chords is overrestrictive — we want passing tones, neighbor tones, and
melodic lines *over an implied harmony*, with the two hands harmonically coherent. The mechanism is a **third global
process** alongside the pitch curve and the accent field.

### The chord track

Introduce $C(t)$ — a slowly varying sequence of **diatonic chords** over the bar grid: the seven diatonic triads
I–vii°, plus the characteristic chords of harmonic/melodic minor. It is a simple Markov chain over chord symbols, with
transition matrix $p(C_{t+1} \mid C_t)$. It does three things at once.

1. **It conditions concrete degree choice without forbidding non-chord tones.** It enters the *same* exponential tilt
   used for substitution (Section 5): chord tones are favoured on strong beats, while passing and neighbor tones are
   permitted — even favoured — on weak beats *between* chord tones. This is "tonal gravity," not "chords only."

2. **It synchronizes the two hands harmonically by shared conditioning, softly.** Both hands tilt toward the *same*
   $C(t)$. At onset coincidences (produced by the sync coupling $h_s$, Section 6), the two hands' concrete degrees are
   both biased to chord tones of the same chord, so the vertical interval is consonant *by construction* — with no chord
   forcing. Where attacks do not coincide, each hand is still governed by the same harmony.

3. **It is fully data-estimable**, which is where Section 1 pays off. Because `anchor_degree` is now stored, a
   lightweight **chord-segmentation pass** runs during extraction: a Viterbi over the seven diatonic triad templates
   (built directly from `SCALE_INTERVALS` and the root) with a self-transition bias, decoding a chord track from each
   exercise's absolute scale-degree content. Tag every figure occurrence with its `(chord, beat-strength)` context
   (the `chord_context` column) and store $p(\text{figure} \mid \text{chord}, \text{beat-strength})$. The non-chord-tone
   behaviour is then **learned from data**, not hand-coded; the only priors injected are the chord vocabulary and the
   triad templates. The chord-transition matrix $p(C_{t+1} \mid C_t)$ falls out of the same decoded tracks and *is* the
   generator's harmonic-progression model.

### Chord representation

Chords use the *same coordinate system as figures* — `(diatonic degree, accidental)` — so borrowed and altered chords
need no special mechanism, and `harmfit` reduces to a membership/role test with no semitone round-trip.

- **Symbol (the Markov/transition layer):** a chord is
  `(root_degree ∈ 1..7, root_accidental ∈ {−1,0,+1}, quality ∈ {maj, min, dim, aug}, extension ∈ {triad, seventh})`,
  all **key-relative**, so the chord track is transposition-invariant — exactly as figures are anchor-relative.
  Inversion / bass degree is an optional later refinement.
- **Expansion to a tone set (for `harmfit` and the Viterbi templates):** generic-third stacking. Member $m$ sits at
  generic degree $d_m = ((r - 1 + 2m) \bmod 7) + 1$, and its accidental is

$$\alpha_m = \big[(\sigma(r) + \alpha_r + q_m) - \sigma(d_m)\big] \bmod 12, \quad \text{taken as a signed residue},$$

  where $\sigma(d)$ is the degree's semitone from `SCALE_INTERVALS` and $(q_0 = 0, q_1, q_2, \dots)$ are the quality's
  semitone intervals (minor triad $= 0, 3, 7$). The result is a set of `(degree, accidental)` tones in the figures'
  coordinate.

**Worked example — minor iv in C major** ($r = 4$, $\alpha_r = 0$, quality minor):

| member | generic degree | natural interval | desired | accidental | note |
| --- | --- | --- | --- | --- | --- |
| root  | 4 | 0 | 0 | 0  | F = (4, 0) |
| third | 6 | 4 (maj 3rd) | 3 | **−1** | A♭ = (6, −1) |
| fifth | 1 | 7 | 7 | 0  | C = (1, 0) |

So $\text{iv} = \{(4, 0), (6, -1), (1, 0)\}$ — the ♭6 emerges automatically as accidental −1, exactly how `NoteToken` /
`FigureDegree` already encode chromatic notes. The same construction yields secondary dominants (V/V → `(4, +1)` = F♯)
and chromatic roots (♭VI → root `(6, −1)`, major quality). One hard limit: `NoteToken` constrains accidental to
`{−1, 0, +1}` (`MIN/MAX_ACCIDENTAL`), so any chord needing a double accidental at some degree is unspellable — a natural
ceiling on the vocabulary, not a practical obstacle.

A window of decoded scale-degree content containing a ♭6 matches the iv template better than IV, so borrowed chords are
detected precisely *because* the enriched database (Section 1) now stores accidentals — closing the loop with the
fact-table refactor.

### Why this over the alternatives

- **Functional-bass / figured-bass** (left hand defines harmony, right hand conditions on it): natural for
  Alberti/block-chord textures, but asymmetric — it breaks down for two-melodic-line writing and forces an ordering on
  per-hand starting-point sampling. Keep it as a *texture mode* (a special case), not the general mechanism.
- **Pitch-class-profile / Krumhansl target histograms**: softer and key-aware, but vaguer than explicit chords and
  weaker for vertical hand coherence. Fold it in as the *emission model* inside the chord Viterbi, not as the primary
  harmony representation.
- **Latent chord track (recommended)** wins because it is the same machinery as everything else (a low-order process +
  the exponential tilt), couples the hands by shared conditioning rather than hard constraint, and is made measurable by
  the enriched database.

### Open decisions

- **Chord resolution**: one chord per beat, per half-bar, or per bar? Finer resolution gives more harmonic motion but
  sparser per-chord figure counts.
- **Template vocabulary scope**: representation already covers borrowed/altered chords (see above), so the open question
  is only *which* to admit as Viterbi templates — the seven diatonic triads alone, or a curated set adding common
  borrowed/applied chords (iv, ♭VI, ♭VII, V/V, the characteristic V / vii° of harmonic minor). Larger vocabularies
  detect richer harmony but risk over-fitting sparse windows; the transition matrix is learned regardless.
- **Seventh chords and inversions**: the `extension` field already admits sevenths; inversion/bass is deferred to a
  later refinement (the left-hand register curve handles bass placement adequately for v1).

## 5. The substitution step — the "stay near reference" guarantee

The figure marginal per group is reproduced *exactly in expectation* only if you sample i.i.d. from the empirical
counts. Every conditioning signal (curve slope, accent, harmony) is a **reweighting** that pulls you off that marginal.
So make conditioning a *soft tilt*, not a hard filter:

$$p(\text{fig}, \text{anchor} \mid \text{group}, \text{curve}, \text{accent}, C) \;\propto\; \underbrace{p_{\text{emp}}(\text{fig} \mid \text{group})}_{\text{your counts}} \cdot \exp\!\Big(\lambda_{\text{curve}}\,\mathrm{slopefit} + \lambda_{\text{harm}}\,\mathrm{harmfit}(\text{fig}, \text{anchor}, C(t), \text{beat-strength})\Big),$$

where $\mathrm{slopefit}$ compares the figure's net contour ($\sum$ relative steps) to the local curve slope, and
$\mathrm{harmfit}$ scores the figure's concrete pitches against the current chord and metrical position (chord tones on
strong beats; passing/neighbor tones permitted on weak beats).

This is an **exponential-family tilt**: the tilted distribution is the **I-projection** of the empirical figure
distribution onto the constraint set — i.e. the *closest distribution in KL to the reference* that still respects the
curve and harmony. It therefore provably minimizes divergence from the reference subject to the control. Each
$\lambda$ is an independent stability ↔ fidelity dial:

- $\lambda_\bullet \to 0$: recover the exact reference marginal (maximum similarity, no control);
- $\lambda_\bullet \to \infty$: rigid curve/harmony following (maximum control, drifts from reference).

The empirical side reuses `commonness_bias` (`musak_model/synthetic/figures.py`), itself a power-tilt
$\mathrm{count}^{\beta}$.

> `commonness_bias` ($\beta$) and the substitution tilts ($\lambda_{\text{curve}}$, $\lambda_{\text{harm}}$) are
> **distinct**: $\beta$ flattens/sharpens the empirical figure frequencies (the "commonality" input), while the
> $\lambda$'s tilt toward curve and harmony compatibility. Keep them separate so the style inputs stay orthogonal.

## 6. Hand interaction: three orthogonal couplings

The two hands interact at three different layers of the model, and these must be kept separate — conflating them causes
trouble. Each is a distinct style input.

| coupling | layer | controls | extremes |
| --- | --- | --- | --- |
| **co-activity** $h_o$ | activity gate | do the hands' active spans overlap? | 0: strict alternation (one rests while the other plays); 1: no onset ever lands on the other hand's rest |
| **sync** $h_s$ | onset placement | within overlapping spans, do attacks coincide? | 0: interleaved attacks; 1: aligned attacks |
| **harmony** | pitch | are concurrent pitches consonant? | shared chord track $C(t)$ (Section 4) |

### Co-activity ($h_o$)

Co-activity acts on the **activity gate** of the point process (Section 3). Give each hand a coarse-resolution (beat or
half-bar) binary gate $A_i(t) \in \{0, 1\}$; onsets for hand $i$ are drawn only where $A_i = 1$. Couple the two gates
with a **Gaussian copula**: draw

$$(z_L, z_R) \sim \mathcal{N}\!\left(0, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}\right), \qquad A_i = \mathbb{1}\!\left[z_i > \Phi^{-1}(1 - h_a^i)\right],$$

so each gate's marginal active probability is exactly the per-hand activity $h_a^i$, and the co-activity is set by

$$h_o \in [0, 1] \;\longmapsto\; \rho = 2 h_o - 1 \in [-1, +1].$$

- $\rho \to -1$: gates anti-correlated → disjoint active spans → **hocket / alternation** ($h_o = 0$).
- $\rho \to +1$: gates identical → no onset ever lands on the other hand's rest ($h_o = 1$).
- $\rho = 0$: independent hands.

The copula is the right tool because it delivers the *prescribed per-hand marginal activity* **and** controllable
co-activity in one object. Two footnotes: (i) perfect anti-correlation is feasible only when
$h_a^L + h_a^R \le 1$ (Fréchet bound); otherwise the copula degrades gracefully to "as disjoint as possible";
(ii) making the gate latent a thresholded *smooth* GP (the same engine as the pitch curve) yields auto-correlated rest
regions — phrasing of rests rather than salt-and-pepper silences.

### Sync ($h_s$) and harmony

**Sync** operates one layer down: given both hands active, $h_s$ is the probability that their onsets coincide on the
grid. Model it as a shared onset mask drawn with weight $h_s$ plus independent masks with weight $1 - h_s$, or as a boost
to $\lambda_L$ at cells where the right hand attacks (and vice versa). **Harmony** operates on pitch and is handled by
the shared chord track (Section 4): it is what makes the coincident attacks produced by $h_s$ sound consonant.

## 7. Hard constraints: keep them out of the stochastic model

The repo already has a complete validator/repair engine in `musak_model/generation/constraints.py`
(`GenerationConstraintState`), enforcing max gap, onset/static span, notes-per-hand, and bar filling. Keep the global
processes soft and use that engine as the **hard gate**: curves + chord track → figure tiling → token sequence → run
through `state.allows(...)` as a rejection/resample filter on each figure. This avoids contorting the probabilistic
model with hard bounds and reuses tested code.

## 8. Fitting and validation loop (per scale_type, hand)

1. **Pitch**: from training onset runs, compute register spread + lag-1 autocorrelation (→ OU $\theta, \sigma$) and the
   low-frequency PSD of length-normalized trajectories (→ arch basis). About 4 numbers.
2. **Accent**: strong/weak rate → metrical ratio; density per beat → $\lambda_0$; grid denominators → grid resolution.
   These *are* the measured statistics, so matching is direct.
3. **Co-activity**: from the decoded activity gates, estimate per-hand activity $h_a^i$ and the empirical gate
   correlation to anchor the $h_o \mapsto \rho$ map.
4. **Harmony**: decode chord tracks (Viterbi over diatonic templates), then estimate the chord-transition matrix
   $p(C_{t+1} \mid C_t)$ and the conditional $p(\text{figure} \mid \text{chord}, \text{beat-strength})$ from the enriched
   fact table.
5. **Tilts $\lambda_{\text{curve}}, \lambda_{\text{harm}}$**: calibrate by a small sweep — generate, measure mean TV
   distance with the existing `figure_distribution_metrics`, and pick the largest $\lambda$'s that keep TV below a target
   (say 0.1). Consider adding a **harmonic-consonance rate at coincident onsets** as a companion validation metric so the
   harmony tilt is checked, not just the figure marginal.

The existing TV-distance and profile-comparison metrics become the generator's tuning objective, closing the loop and
providing the empirical guarantee of "near reference."

## Rejected alternatives

- **Flat HMM / Markov chain over absolute pitch+rhythm**: conflates register drift with local contour, hides the style
  knobs, and wanders out of register. Strictly worse here *because* the figure dataclasses already encode the local
  structure — the global model can and should stay low-order.
- **Neural autoregressive / diffusion**: that is what `musak_model` (the LLM) already is. The point of this generator is
  to be the cheap, controllable, interpretable classical complement.
- **Forcing chords for harmony**: overrestrictive — it eliminates passing and neighbor tones. The latent chord track +
  soft harmonic tilt (Section 4) gives tonal coherence while preserving melodic freedom.
- **Functional-bass as the sole harmony mechanism**: asymmetric and weak for two-melodic-line textures; retained only as
  an optional texture mode.

## Summary recommendation

The generator is **three coupled global processes** feeding one figure-substitution step:

- **Pitch** = a random low-frequency arch (DCT/spline coefficients from the empirical PSD) + a mean-reverting **OU**
  residual on the diatonic lattice (a GP with $k_{\text{arch}} + k_{\text{Matérn-½}}$). Mean reversion keeps it in
  playable register; both parts are $O(N)$ and each parameter is a moment read off the data.
- **Accents** = a **log-Gaussian Cox process**: a smooth activity envelope (same engine as the pitch curve) × metrical
  indispensability, with intensity doubling as the accent weight, plus a coarse activity gate.
- **Harmony** = a **latent diatonic chord track** $C(t)$, a Markov chain over diatonic triads, decoded from training data
  by Viterbi segmentation and entering substitution as a soft harmonic tilt.

These feed an **exponential tilt**
$p_{\text{emp}}(\text{fig}) \cdot \exp(\lambda_{\text{curve}}\,\mathrm{slopefit} + \lambda_{\text{harm}}\,\mathrm{harmfit})$,
the minimum-KL-to-reference distribution subject to following the curve and harmony; the $\lambda$'s are the
stability ↔ fidelity dials, kept orthogonal to `commonness_bias`.

The two hands interact through **three orthogonal couplings**: co-activity $h_o$ (Gaussian-copula activity gates),
sync $h_s$ (onset coincidence), and harmony (shared chord track).

Hard playability stays out of the stochastic model — `constraints.py` is the reject/repair gate. The **reference
database** is refactored into a rich occurrence-level fact table, from which the existing relative-figure counts (and the
TV-distance metric) remain a materialized projection.

The decisive structural point: `FigureNGram` is anchor-relative and rhythm-normalized, so it already owns the high-order
local structure. That is exactly why the global processes can be low-order, why harmony enters as conditioning rather
than as a vocabulary change, and why this design beats a flat Markov chain over absolute notes.

## Implementation

The design above is realised in six incremental phases, each independently verifiable. Generator-specific logic lives
under `musak_model/synthetic/`; figure-extraction and database changes live in their owner
(`musak_model/n_grams/profile/...`, `musak_model/training/stages/...`). The dependency direction is one-way:
`synthetic` may depend on `n_grams` / `decoder`, never the reverse.

### Decisions

- **Persistence**: promote the SQLite store from an intermediate `work.sqlite3` to the durable, queryable reference
  artifact; CSV/profile become *exported projections* so the existing TV-distance metric and notebooks keep working.
- **Chord resolution**: configurable (beat / half-bar / bar); the default is chosen later after inspecting decoded
  tracks.
- **Harmony in v1**: the `n_grams` DB does **not** store chord context; chord-conditioned figure statistics are computed
  synthetic-side by reusing the shared counting routine. If harmony matures, the chord module moves to a shared package
  and chord context folds into extraction.

### Conventions (per `docs/guidelines.md`)

Frozen Pydantic models for validated/serialized data (chord symbol, configs); dataclasses only for small internal
state; no abbreviated names; `Final` for constants; keyword-only options after `*`; `match` over `isinstance` chains;
full type hints validated by `mypy`; tests mirror the package layout under `tests/musak_model/...`, parametrized with
test-case dataclasses and shared fixtures; search for existing logic before adding helpers.

### Phase 1 — Chord representation module (first deliverable)

A pure, dependency-light chord primitive + YAML vocabulary, expandable to a diatonic tone set. New package
`musak_model/synthetic/harmony/`:

- `schema.py` — `ChordQuality` (`MAJOR/MINOR/DIMINISHED/AUGMENTED`) and `ChordExtension`
  (`TRIAD/SEVENTH/NINTH/ELEVENTH`, plus altered variants like `FLAT_NINTH`, `SHARP_ELEVENTH`) `StrEnum`s; a frozen
  Pydantic `Chord` model `(root_degree: 1..7, root_accidental: −1..1, quality, extension)`, key-relative.
- `vocabulary.py` — `ChordVocabularyConfig` (frozen Pydantic, `.load(path)` mirroring
  `musak_model/n_grams/config.py:NGramAnalysisConfig.load`): the quality→semitone-interval table and per-quality /
  per-extension enable flags. Mirror the structure of `musak/config/inversions.yml` (`chords_definitions` + on/off
  `default_settings`) but in scale-degree terms; v1 enables triads of the four core qualities only.
- `expansion.py` — `expand_chord_to_tones(chord, *, scale_type) -> tuple[ChordTone, ...]` with
  `ChordTone = (degree, accidental)`, implementing the generic-third stacking of Section 4.

Config at `musak_model/configs/generation/chords.yml`; `CHORD_VOCABULARY_CONFIG_PATH` in `musak_model/paths.py`.
Reuse `SCALE_INTERVALS` / `ScaleType` / `MIN/MAX_ACCIDENTAL` (`tokens/schema.py`), `load_yaml_config`
(`musak_shared/files.py`), `scale_size_for_type` (`n_grams/figure/builder.py`). **Verify** with parametrized cases
(minor iv → `{(4,0),(6,−1),(1,0)}`; V/V → `(4,+1)`; ♭VI root `(6,−1)`; diatonic triads across all three scales;
sevenths when enabled; an unspellable double-accidental raising), plus `mypy` and pre-commit.

### Phase 2 — Reference database enrichment + unification

Store the discarded context, make SQLite durable, and remove the orphaned duplicate counting path; CSV/profile outputs
stay byte-compatible as projections.

- **Enrich the key** (`n_grams/profile/streaming/schema.py:FigureCountKey`,
  `n_grams/profile/streaming/tables.py:_COUNTS_TABLE`): add `anchor_degree`, `anchor_accidental`, `base_duration`
  (string Fraction), `metrical_position`, `time_signature` to the primary key. Use a coarse `metrical_position`
  descriptor (beat index / indispensability level), not a raw Fraction, to bound cardinality. `signature.py` already
  computes the anchor and shortest duration — expose them instead of discarding; `streaming/counting.py` populates the
  new fields from `HandOnsetRun` / `PitchedOnset` / `SegmentMetadata`.
- **Promote SQLite to durable** (`store.py`, `export.py`, `artifacts.py`): rename the durable DB (e.g.
  `figures.sqlite3`), stop deleting it in `clear_figure_work`, keep resume/state-key logic. `export_counts_csv` /
  `profile_from_store` aggregate (`GROUP BY scale_type, hand, n, figure`) so `counts.parquet` / `profile.json` are
  unchanged. Add a thin read API (`FigureReferenceStore`) exposing marginal/conditional queries for the generator.
- **Unify counting**: `analyze-n-grams` and `pretrain` already share the single `count_sample_figure_signatures`
  worker, so both now write the enriched schema with no further work. Deleting the parallel
  `n_grams/figure/samples/{counter,batches,merge,single}.py` is **deferred** to a separate change: `merge_scale_counts`
  (from `samples/merge.py`) is a live dependency of `evaluation/generation/figure_metrics.py`, so migrating that
  consumer onto the streaming path is its own evaluation-focused task. `samples/schema.py` stays regardless (its
  `FigureNGramCountsByScale` is used by `profile/io.py` and `synthetic/figures.py`).

**Verify**: existing `test_io.py` round-trips pass; new tests assert the enriched key round-trips and the CSV projection
equals the pre-change CSV; `make analyze-n-grams` leaves `counts.parquet`/`profile.json` unchanged while the `.db` gains
columns; a tiny `make pretrain` leaves figure metrics unchanged. Move deleted modules' tests in the same change.

### Phase 3 — Chord decoding from data

Turn implied harmony into a chord track and a chord-conditioned figure distribution, with deliberately simple, pluggable
machinery. New `musak_model/synthetic/harmony/decoding.py`:

- `ChordDecoderConfig` (frozen Pydantic, YAML): resolution (`BEAT/HALF_BAR/BAR`), self-transition bias, candidate
  templates from the Phase 1 vocabulary.
- Window each segment via `segment_to_piano_roll_events` (`decoder/piano_roll.py`); per window, score each template by
  coverage vs metrically weighted non-chord-tone penalty; decode by **Viterbi** (pluggable `ChordDecoder` protocol).
- Build `p(figure | chord, beat-strength)` synthetic-side by iterating segments once, jointly calling the shared
  figure-occurrence routine (extended in Phase 2 to yield occurrence position) and the decoder, into a synthetic-owned
  store.

Reuse `segment_to_piano_roll_events` / `PianoRollEvent`, `pitch_to_degree` (`data/converter.py`),
`note_token_to_midi_pitch` (`tokens/pitch.py`), Phase 1 `expand_chord_to_tones`. **Verify** on synthetic windows
(I–IV–V–I decodes correctly; passing-tone lines still resolve to the right triads) and confirm `p(figure | chord)`
marginalizes back to the unconditioned figure distribution.

### Phase 4 — Global stochastic processes

New modules under `musak_model/synthetic/processes/`, each estimated from the enriched DB:

- `pitch.py` — register curve (Section 2): random low-frequency arch + mean-reverting OU residual on the diatonic
  lattice; fit register spread, lag-1 autocorrelation, and low-frequency spectrum per `(scale_type, hand)`;
  `HAND_HOME_OCTAVES` as the OU mean.
- `accent.py` — LGCP (Section 3): metrical indispensability × smooth activity envelope; estimate ratios/density from the
  rhythm tables.
- `chord_track.py` — Markov sampler over diatonic chords using Phase 3's transition matrix; resolution from config.

**Verify** with property tests: sampled trajectories match target spread/autocorrelation; accent density and strong/weak
ratio match measured values; chord-track stationary distribution matches its transition matrix.

### Phase 5 — Figure substitution + assembly

New `musak_model/synthetic/substitution.py` + `assembly.py`:

- Exponential tilt `p_emp(fig) · exp(λ_curve·slopefit + λ_harm·harmfit)` (Section 5) over enriched-vocabulary
  candidates, reusing `commonness_bias` (`synthetic/figures.py:FigureVocabulary.sample`) as the `p_emp` side;
  `slopefit` from the pitch curve, `harmfit` from Phase 1 tones vs the chord track.
- Tile figures along the curve/grid (anchor = curve value, base duration = sampled), emit `Token`s per hand, and gate
  each prefix through `GenerationConstraintState` (`generation/constraints.py`) as a reject/resample filter.

**Verify**: generated sequences decode to valid `Segment`s passing all constraints; a scratch script renders exercises
and round-trips through the decoder.

### Phase 6 — Style inputs, hand couplings, calibration

- The three orthogonal couplings (Section 6): co-activity `h_o` (Gaussian-copula activity gates, `ρ = 2h_o − 1`,
  marginals `h_a^i`), sync `h_s`, harmony (shared chord track), in `synthetic/processes/` + `synthetic/style.py`.
- Calibrate `λ_curve`, `λ_harm` by sweeping and measuring mean TV distance via `figure_distribution_metrics`
  (`n_grams/profile/metrics/distribution.py`); add a harmonic-consonance-at-coincident-onsets companion metric.

**Verify**: the generated corpus scored by `figure_distribution_metrics` and `figure_profile_comparison_metrics` stays
under the TV target, and the co-activity/sync extremes produce the expected alternation vs full-overlap behaviour.

