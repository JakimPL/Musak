# Synthetic Exercise Generator: Design

This document describes a classical, stochastic, data-driven generator of sight-reading piano exercises. Its purpose is
to produce synthetic data that complements the LLM-based `musak_model`, while remaining controllable, interpretable, and
faithful to the figure statistics of the training reference.

The generator is intentionally *not* a neural model. It separates a small set of **low-order global stochastic
processes** (register trajectory, rhythmic activity) from the **empirical figure vocabulary** that already encodes local
micro-structure. The design exploits three facts that the existing codebase already establishes.

## What the data model already determines

1. **Figures are purely relative.** `build_figure_ngram` (`musak_model/n_grams/figure/builder.py`) anchors pitch to the
   minimum diatonic position of onset 0 and normalizes durations to the shortest onset (`_normalize_durations`). A
   `FigureNGram` is therefore a *translation-invariant contour + scale-invariant rhythm template* — it encodes **micro**
   structure (local shape and relative rhythm) and nothing about absolute register or tempo.

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
(`musak_model/synthetic/figures.py`) already samples them. So the global model only supplies *register trajectory* and
*rhythmic activity envelope*. This is why the approach beats a flat Markov chain over absolute notes (see
[Rejected alternatives](#rejected-alternatives)).

## 1. The pitch curve

### Unifying frame: a Gaussian process on the diatonic lattice

Model each hand's onset-pitch trajectory $P_i(t)$ as

$$P_i(t) \sim \mathcal{GP}\big(\mu_i,\; k(t,t')\big), \qquad k = k_{\text{arch}} + k_{\text{wobble}},$$

with $\mu_i$ = the hand's home register (the code already defines `RIGHT_HAND_HOME_OCTAVE = 5`,
`LEFT_HAND_HOME_OCTAVE = 3`), $k_{\text{arch}}$ a long-lengthscale kernel (the slow phrase arch), and $k_{\text{wobble}}$
a short-lengthscale Matérn-½ kernel (local register drift). Conditioning on a concrete start point is GP posterior
conditioning on $P_i(0) = p_0$; conditioning on octave only means conditioning on a band. Sample continuous, then
quantize to the lattice at substitution time.

Every concrete option below is a special case of this GP — which is the honest way to see them.

#### Option A — Ornstein–Uhlenbeck / AR(1) (the cheap $k_{\text{wobble}}$)

$$dP_i = \theta(\mu_i - P_i)\,dt + \sigma\,dW \quad\Longleftrightarrow\quad P_{k+1} = P_k + \theta(\mu_i - P_k) + \varepsilon_k.$$

Stationary variance is $\sigma^2 / 2\theta$ (register spread); autocorrelation is $\rho(\tau) = e^{-\theta\tau}$. Fit
$\theta, \sigma$ from two measured numbers: the empirical spread of onset pitch per hand and its lag-1 autocorrelation.
Mean reversion is the feature — it keeps the hand inside playable register instead of drifting out, which a plain
GP/random walk will not. Bounding $\varepsilon_k$ gives the "max melodic gap" softly. Cost is $O(N)$, with 3
interpretable parameters.

#### Option B — Band-limited random function (the cheap $k_{\text{arch}}$)

$$P_i(t) = m_i(t) + \sum_{j=1}^{J} a_j\,\phi_j(t), \qquad a_j \sim \mathcal{N}\big(0,\, S(f_j)\big),$$

with $\phi_j$ a **DCT/B-spline basis** (not plain sines — this avoids wrap-around at the boundaries) and $S(f_j)$ the
empirical power spectral density of length-normalized training trajectories (Welch-average periodograms after resampling
each exercise to unit length). Small $J$ gives guaranteed smoothness and a *fixed, stable* generator that you fully
control. This is exactly "smooth movement through the entire exercise" = low-pass noise.

#### Recommendation: arch + OU

Use **B around a mean, with A as the residual**: a random low-frequency arch (capturing the global shape *and its
variability* from data) plus an OU wobble that is mean-reverting and lag-1-matched. In GP language this is precisely
$k_{\text{arch}} + k_{\text{Matérn-½}}$; implement it as arch-spline + OU because both are $O(N)$ and each parameter is a
moment you can read directly off the data.

| pitch model        | matches              | smooth      | stays in register   | discrete-native | params            |
| ------------------ | -------------------- | ----------- | ------------------- | --------------- | ----------------- |
| full GP            | autocovariance       | yes         | needs a cap         | quantize        | kernel hyperparams |
| OU / AR(1)         | spread + lag-1       | rough (C⁰)  | yes (mean-revert)   | easy            | θ, σ, μ           |
| DCT/spline random  | spectrum             | yes (band)  | soft                | quantize        | #bands, gains     |
| **arch + OU (rec.)** | spectrum + lag-1     | yes         | yes                 | quantize        | few               |

The curve's **value** sets the anchor (the absolute diatonic position to drop a figure on); the curve's **local slope**
biases *which* figure (ascending vs descending). That is the "figure substitution along the curve," made precise — and
it exploits the fact that figures are anchor-relative.

## 2. The accent grid

This is a **marked point process on the duration grid**: each grid cell is onset / no-onset, and onsets carry an accent
weight. The rhythm extractor (`musak_model/n_grams/profile/rhythm/extraction.py`) already measures the strong-vs-weak
onset rate, grid-alignment rate per denominator, onset density, and IOI/duration n-grams. Those are exactly the moments
to match.

#### Option A — Inhomogeneous Bernoulli with a metrical intensity

$$\Pr[\text{onset at } t] = \sigma\big(\beta_0 + \beta_1 \cdot \mathrm{ind}(t)\big), \qquad \lambda(t) = \lambda_0 \cdot \mathrm{ind}(t)^{\gamma},$$

where $\mathrm{ind}(t)$ is a **metrical-hierarchy weight** (Barlow indispensability from the meter's prime factorization,
or simply per-subdivision-level probabilities). $\sum_t \lambda(t)$ is the expected density (the style input $h_d$); the
strong/weak ratio you measure *is* $\lambda_{\text{strong}} / \lambda_{\text{weak}}$, so it is direct moment-matching.
The downside: independent cells ignore sequential rhythm.

#### Option B — Rhythm-figure n-gram chain

Sample a Markov chain over the already-extracted IOI/duration n-grams. This matches the rhythm-n-gram reference by
construction and captures runs (sequential correlation), but it produces a discrete token sequence rather than "weighted
impulses," and needs metrical conditioning.

#### Option C — Log-Gaussian Cox process (recommended)

$$\log \lambda_i(t) = \underbrace{\mathrm{ind}(t)}_{\text{meter}} + \underbrace{g_i(t)}_{\text{smooth activity envelope}},$$

where $g_i(t)$ comes from the *same* low-frequency machinery as the pitch curve. This is the "weighted impulses"
picture: onsets are a Bernoulli/thinned draw with probability $\propto \lambda$, and the **accent weight is $\lambda$ at
that cell**, used downstream to bias figure choice (strong → longer/denser figure or chord; weak → passing note). The
smooth $g_i$ produces **phrasing** — auto-correlated busy/sparse regions — which independent Bernoulli cannot. The style
knobs map cleanly: $h_a^i$ = mean of $g_i$, $h_d^i$ = gain on $\lambda$.

Let the **figures carry IOI micro-structure**: you do not need Option B to match rhythm n-grams if substitution respects
figure rhythm. Use Option C for the skeleton and accents, and let figure rhythm fill the detail.

## 3. The substitution step — the "stay near reference" guarantee

This is the crux. The figure marginal per group is reproduced *exactly in expectation* only if you sample i.i.d. from
the empirical counts. Every conditioning signal (match the curve slope, match the accent) is a **reweighting** that
pulls you off that marginal. So make conditioning a *soft tilt*, not a hard filter:

$$p(\text{fig} \mid \text{group}, \text{curve}, \text{accent}) \;\propto\; \underbrace{p_{\text{emp}}(\text{fig} \mid \text{group})}_{\text{your counts}} \cdot \exp\!\big(\lambda \, \mathrm{compat}(\text{fig}, \text{curve}, \text{accent})\big),$$

where $\mathrm{compat}$ scores, for example, (a) figure net contour ($\sum$ relative steps) vs local curve slope, and
(b) figure rhythmic signature vs local accent/IOI target.

This is an **exponential-family tilt**, and it has the exact property required: the tilted distribution is the
**I-projection** of the empirical figure distribution onto the "follows-the-curve" constraint set — i.e. it is the
*closest distribution in KL to the reference* that still respects the curve. It therefore provably minimizes divergence
from the reference subject to the control. The scalar $\lambda$ **is** the stability ↔ data-fidelity knob:

- $\lambda \to 0$: exact reference marginal (maximum similarity, no curve control);
- $\lambda \to \infty$: rigid curve-following (maximum control, drifts from reference).

This is the precise formalization of "data-based yet stable, close to reference," and it reuses `commonness_bias`
(`musak_model/synthetic/figures.py`), itself a power-tilt $\mathrm{count}^{\beta}$, as the $p_{\text{emp}}$ side.

> `commonness_bias` and $\lambda$ are **two distinct tilts**: $\beta$ flattens/sharpens the empirical figure frequencies
> (the "commonality" input), while $\lambda$ tilts toward curve-compatibility. Keep them separate so the style inputs
> stay orthogonal.

## 4. Hand coupling (the style inputs $h_s$, $h_a$, $h_d$)

- **Pitch**: a 2-D GP/OU with cross-covariance $\rho$ between hands. $\rho > 0$ gives parallel motion, $\rho < 0$
  contrary motion; $\rho$ is a style knob.
- **Sync ($h_s$)**: draw a *shared* onset mask with weight $h_s$ and independent masks with weight $1 - h_s$; or boost
  $\lambda_L$ at cells where the right hand attacks (and vice versa) by $\gamma h_s$. This directly controls attack
  coincidence.
- **Activity / density**: $h_a^i = \mathbb{E}[g_i]$ and $h_d^i$ = gain on $\lambda_i$ fall straight out of Section 2.

## 5. Hard constraints: keep them out of the stochastic model

The repo already has a complete validator/repair engine in `musak_model/generation/constraints.py`
(`GenerationConstraintState`), enforcing max gap, onset/static span, notes-per-hand, and bar filling. Keep the curve
model soft and use that engine as the **hard gate**: curves → figure tiling → token sequence → run through
`state.allows(...)` as a rejection/resample filter on each figure. This avoids contorting the probabilistic model with
hard bounds and reuses tested code.

## 6. Fitting and validation loop (per scale_type, hand)

1. **Pitch**: from training onset runs, compute register spread + lag-1 autocorrelation (→ OU $\theta, \sigma$) and the
   low-frequency PSD of length-normalized trajectories (→ arch basis). About 4 numbers.
2. **Accent**: strong/weak rate → metrical ratio; density per beat → $\lambda_0$; grid denominators → grid resolution.
   These *are* the measured statistics, so matching is direct.
3. **Tilt $\lambda$**: calibrate by a small sweep — generate, measure mean TV distance with the existing
   `figure_distribution_metrics`, and pick the largest $\lambda$ that keeps TV below a target (say 0.1). The existing
   TV-distance and profile-comparison metrics become the generator's objective, closing the loop and providing the
   empirical guarantee of "near reference."

## Rejected alternatives

- **Flat HMM / Markov chain over absolute pitch+rhythm**: the "obvious" classical move, but it conflates register drift
  with local contour, hides the style knobs, and wanders out of register. It is strictly worse here *because* the figure
  dataclasses already encode the local structure — the global model can and should stay low-order.
- **Neural autoregressive / diffusion**: that is what `musak_model` (the LLM) already is. The point of this generator is
  to be the cheap, controllable, interpretable classical complement.

## Summary recommendation

Per `(scale_type, hand)`:

- **Pitch** = a random low-frequency arch (DCT/spline coefficients drawn from the empirical PSD) + a mean-reverting
  **OU** residual, on the diatonic lattice — formally a GP with $k_{\text{arch}} + k_{\text{Matérn-½}}$. Mean reversion
  keeps the trajectory in playable register; both parts are $O(N)$ and each parameter is a moment read off the data.
- **Accents** = a **log-Gaussian Cox process**: a smooth activity envelope (the same engine as the pitch curve) ×
  metrical indispensability, with the intensity doubling as the accent weight. This matches the already-measured
  strong/weak and density statistics directly, and phrasing emerges from the envelope's autocorrelation.
- **Substitution** = an **exponential tilt** $p_{\text{emp}}(\text{fig}) \cdot e^{\lambda\,\mathrm{compat}}$, which is the
  minimum-KL-to-reference distribution subject to following the curve. $\lambda$ is the single stability ↔ fidelity
  dial, kept orthogonal to the existing `commonness_bias`.
- **Hard constraints** stay out of the stochastic model — reuse `constraints.py` as a reject/repair gate.
- **Validation** loops on the existing TV-distance metric (`distribution.py`), which becomes the generator's tuning
  objective.

The decisive structural point: `FigureNGram` is anchor-relative and rhythm-normalized, so it already owns the high-order
local structure. That is exactly why the global processes can be low-order, and why this design beats a flat Markov
chain over absolute notes.
