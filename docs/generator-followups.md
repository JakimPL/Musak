# Gaps between `docs/generator.md` and the current implementation

`docs/generator.md` is the design document. This file is the running list of places where the code does
not yet match it — gaps to be closed. Each entry names what the design says, what the code does today, and
what closing the gap requires.

The entries fall into two kinds: **unimplemented design** (the feature simply is not there yet) and
**code diverges from design** (a step exists but computes something different from the doc). The latter were
surfaced by the code-derived model in [`docs/generator-model.md`](generator-model.md); its §6 carries the
`D1`–`D10` labels cross-referenced below.

Already closed: the accent field is wired into substitution, the length-0 decoder window is guarded, and
the activity gate plus register/accent reads are now per grid cell (gaps 1 and 3). (The register home-offset
$\mu_i$ was never a gap — `octave_offset` is home-relative and $\mu_i$ is applied at token-to-MIDI
conversion; see `docs/generator.md` §3.) A quick-wins pass then closed gaps 5, 6, 12 and 13, and a
structural pass closed gaps 2 (sync coupling) and 9 (sub-bar harmonic rhythm) (all below).

A musical pass then closed #4 (metrical harmonic conditioning) and #8 (texture mode), and landed the
**functional-harmony interim** of #11 (a hand-authored functional diatonic chord prior, now the generation
default) plus the Stage 0 musical-evaluation metrics (`musical_metrics`, wired into `GenerationSuiteEvaluator`
→ MLflow). A fitting pass then closed #10 (register) and the accent half of #11: both are now fit from
**persisted corpus sufficient statistics** (computed once in the figure-profile corpus pass, not recomputed
on the fly) and loaded into generation as per-`(scale_type, hand)` overrides via a `FittedGeneratorConfig`
artifact. The loop is wired end-to-end: `scripts/fit_generator.py` (`make fit-generator`) reads the persisted
statistics, writes `fitted_generator.json` next to the figure vocabulary (`all/`), and generation loads it
(the notebook surfaces the fit status).

A harmony pass then closed the **empirical chord loop** of #11: the corpus is Viterbi-decoded in the figure
pass into a persisted `chord/` sub-store (transition counts + figure-by-chord counts), `fit_generator.py`
bakes the empirical (functional-prior-smoothed) per-scale-type transition model and $p(\text{figure} \mid C)$
into `fitted_generator.json`, and generation consumes them via a `chord_model` selector and the
`lambda_chord_figure` tilt. The chord decoder was relocated to a neutral `musak_model/harmony/` package so
the figure pass can decode without inverting the n-grams↔synthetic layering.

**Still open:** #7 (figure-length distribution) — plus the greedy-fill note at the end. Three items are
**deferred by design**: the calibration↔harmony/texture coupling (needs a proper mathematical model),
figure-to-figure continuity (novel approach), and **envelope fitting** (the accent envelope's amplitude/decay
need a bar-to-bar occupancy variance/autocorrelation statistic not currently stored; see #11).

---

## 1. Activity gating is bar-resolution, not grid-cell-resolution — CLOSED

**Design (§4, §7).** The accent field is a marked point process on the bar grid; each grid cell is
independently an onset or a rest, and the hand-coupling gate acts per cell, so a hand can fall silent for
part of a bar.

**Resolution.** `SegmentGenerator.generate` now takes `grid_count_per_bar` and samples per-cell accent
weights (`AccentFieldSampler.sample_weights`), coupled onset masks, and hand-coupling gates at `cell_count =
bar_count * grid_count_per_bar`. A cell fires iff its onset mask *and* the per-cell coupling gate are active
for the hand; the cursor walks each bar, resting unfired stretches and starting a figure at each fired cell,
so a hand can fall silent mid-bar. Figure durations are unchanged — a figure may still span many cells; the
grid only fixes onset times. (The onset draw itself was later moved into the hand-coupling layer; see #2.)

## 2. Sync coupling is not implemented — CLOSED

**Design (§7).** Hand interaction has three couplings — co-activity, **sync** ($h_s$: the probability that
both hands' attacks coincide on the grid), and shared harmony.

**Code before.** `HandCouplingSampler` modelled only co-activity (the Gaussian-copula gate); the two hands'
onset Bernoulli draws were independent inside `AccentFieldSampler.sample`.

**Resolution.** The per-hand onset draw moved out of `AccentFieldSampler` (now weights-only via
`sample_weights`; `AccentCell` and `sample` were removed) into `HandCouplingSampler.sample_onsets`, which
owns both couplings. With probability `sync_strength` a cell uses one shared uniform for both hands
(comonotonic, so attacks coincide while each hand's marginal onset probability stays equal to its weight);
otherwise the hands draw independently. `generate()` now fires a cell iff its coupled onset mask and the
co-activity gate are both active. Wired through `HandCouplingConfig.sync_strength` (`hand_coupling.yml`),
`SyntheticGenerationRequest`, and a notebook slider.

Note: relocating the onset draw shifts the rng stream for every seed (even at `sync_strength=0`), so
generated token sequences differ from the pre-change output; the determinism tests assert run-to-run
equality and structural validity, not golden tokens, so they are unaffected.

## 3. Register and accent are shared across all figures in a bar — CLOSED

**Design (§3, §6).** The register curve yields an integer diatonic position *per onset step*, and the
accent value entering the substitution tilt is the envelope *at the current cell*.

**Resolution.** The register curve is now sampled at grid resolution (`length =
bar_count * grid_count_per_bar`) and each figure reads its anchor from the firing cell, its slope target
from that cell to the next, and its envelope value from the firing cell's accent weight, so sub-bar
register motion and per-cell accent shaping are modelled.

---

## 4. Harmonic-fit tilt ignores metrical position — `D1` — CLOSED

**Design (§6).** The harmonic-fit score is $H(f, C, m)$: chord tones are rewarded on strong beats and
non-chord tones are permitted on weak beats, at a weight that depends on the figure's metrical position $m$.

**Code before.** `harmonic_fit` returned the plain chord-tone fraction over all note instances; no metrical
weighting.

**Resolution.** `harmonic_fit` now returns a **metrical-strength-weighted** mean of the per-onset chord-tone
fraction: onset $i$ sits at bar position `(metrical_position + i) % grid_count_per_bar` (the one-cell-per-onset
proxy, consistent with the slope-fit span in #5), weighted by its indispensability `gcd(position, M)/M`. Chord
membership therefore matters most on strong cells and barely on weak ones, so non-chord passing tones are
tolerated off the beat. The firing cell's `metrical_position` and `grid_count_per_bar` are threaded from
`_place_one_figure` through `sample_substituted_figure`. At `grid_count_per_bar = 1` it reduces to the plain
chord-tone fraction. (Effective only when `lambda_harmonic > 0`.)

## 5. Slope-fit uses an endpoint proxy compared against a one-cell slope — `D2` — CLOSED

**Design (§6).** $S(f, P_i)$ compares the figure's **net contour** to the slope of the register curve.

**Code before.** `figure_net_contour` returned the lowest note of the figure's **last onset**,
anchor-relative, and `slope_fit = -|figure_net_contour - target_slope|`, where `target_slope` was
`curve[fired+1] - curve[fired]` — a **single-cell** difference. A figure spanning several cells was scored
against a one-cell register change, so the two quantities lived on different scales.

**Resolution.** `_place_one_figure` now computes `target_slope = curve[min(fired + figure_length - 1,
last)] - curve[fired]` — the register change over as many cells as the figure has onsets (one cell per
onset) — so the slope target and the figure's multi-onset displacement share a scale. `figure_net_contour`
is kept as the net displacement of the figure's lowest voice (the appropriate quantity for slope-matching;
the doc's "Σ relative steps" wording was the loose part), documented as such, and `docs/generator.md` §6 is
reconciled to match. The mean-voice alternative for polyphonic onsets was considered and not adopted — the
lowest voice is consistent with the figure anchoring convention.

## 6. Accent-fit ignores chord-tone onsets — `D9` — CLOSED

**Design (§6).** $A(f, \lambda_i)$ compares the figure's internal accent shape (longer notes **and
chord-tone onsets** aligned with the figure's strong points) to the LGCP envelope value at the current cell.

**Code before.** `accent_fit` returned `stress * envelope_value` with
`stress = Σ_i (durationᵢ / total) · (gcd(i, n) / n)` over the onset index `i`. Chord-tone onsets did not
enter the score — only durations did.

**Resolution.** `accent_fit` now blends two normalised emphases at the figure's internal strong points
(`gcd(i, n)/n`): the duration weight and a chord-tone-onset weight, falling back to duration-only when the
figure has no chord tones so the score reduces to the prior rhythmic accent shape. The chord-tone term
reads `anchor`, `scale_type` and the chord pitch-class set (already available at the call site; threaded
through `tilted_log_probabilities`). The onset-index "internal metrical skeleton" is kept deliberately: the
design's envelope $\lambda_i(k)$ is a **per-cell scalar**, so the figure's shape can only be *scaled* by it,
not aligned to it onset-by-onset — the `stress · envelope_value` structure is therefore correct, and
`envelope_value` remaining a flat multiplier is by design. Per-onset bar-grid alignment is deferred because
it depends on the base duration chosen after scoring.

## 7. Figure length is sampled uniformly, not from the empirical distribution — `D7`

**Design.** The figure vocabulary carries corpus statistics; figure length is part of the empirical
distribution being matched.

**Code today.** `_place_one_figure` picks `figure_length = int(rng.choice(self.figure_lengths))`
(`generator.py:387`) — uniform over `[min_n, max_n]`. `FigureVocabulary.length_distribution()` exists but is
never used on the generation path.

**To close the gap.** Sample the length from `length_distribution()` (optionally per scale/hand), or fold
length into a single tilted choice over a combined candidate pool. **Check before changing:** calibration
scores TV distance per `(scale, hand, n)` group independently, so uniform length keeps every `n` populated
for the metric — confirm against `figure_distribution_metrics` whether uniform sampling is intentional.

## 8. Monorhythmic filtering is unused; chord/polyrhythmic figures are admitted — `D8` — CLOSED

**Design (§10).** A monophonic two-melodic-line texture is the default; chordal/Alberti settings are an
*optional* mode.

**Code before.** `_figure_entries_by_group` admitted *all* figures in each `(hand, n)` group, including
`chords_only` and polyrhythmic ones, scattering chords into the melodic lines.

**Resolution.** `SubstitutionConfig` gains a `monophonic: bool`; when set, `_figure_entries_by_group` filters
the candidate pool to `entry.figure.monophonic`. Generation/notebook defaults to monophonic (a UI checkbox);
**calibration explicitly sets `monophonic=False`** (unfiltered) to keep the figure-shape TV metric fair until
the deferred calibration↔texture coupling is designed.

## 9. Harmonic rhythm is fixed at one chord per bar — `D3` — CLOSED

**Design (§5).** The harmonic-rhythm resolution is a configurable power-of-two note value (whole/half/
quarter), bar-aligned with truncation, yielding sub-bar chord windows in odd meters.

**Code before.** Generation sampled `chord_track` with `length=bar_count` and computed the chord pitch
classes once per bar; sub-bar windowing existed only on the offline `ChordDecoderConfig.resolution`.

**Resolution.** The bar-aligned, barline-truncated tiling is now a shared `chord_window_grid`
(`synthetic/harmony/windows.py`), used by both `decoding/windows.py:sounding_windows` and the generator.
`SegmentGenerator.generate` takes a power-of-two `chord_resolution`, samples one chord per window
(`chord_track` length = window count), and conditions each fired cell by the chord at its first onset via a
precomputed per-cell pitch-class map (`bisect` over the window starts). Wired through `CalibrationConfig`
(default 1, also in `calibration.yml`), `SyntheticGenerationRequest`, and a notebook control. No
divisibility constraint ties `chord_resolution` to `grid_count_per_bar`.

Note: `chord_resolution=1` reproduces one-chord-per-bar only in meters no longer than a whole note (the 4/4
default and common meters ≤ 4/4). In 5/4 and larger it yields the bar-aligned-truncated windows the design
prescribes (a whole-note window plus a tail) — more than one window per bar, by design.

## 10. Register curve is hand- and scale-agnostic — `D4` — CLOSED

**Design (§3, §9).** The OU parameters $(\theta_i, \sigma_i)$ and arch amplitudes $\{A_j\}$ are per
`(scale_type, hand)` — fit from the empirical register spread, lag-1 autocorrelation, and low-frequency PSD.

**Code before.** `RegisterCurveSampler.sample` discarded `scale_type`/`hand`; both hands drew i.i.d. from one
shared `RegisterCurveConfig`.

**Resolution.** `RegisterCurveSampler` now holds `config` (default) + `overrides: tuple[RegisterCurveOverride,
...]` and selects per `(scale_type, hand)` (default fallback). The moments are **fit from persisted corpus
sufficient statistics, not by walking segments at fit time**: the figure-profile corpus pass accumulates
per-`(scale_type, hand)` running sums (`n_grams/profile/register/` — Σtrend², Σresidual², Σresidual·lag, n),
partitioning each onset-register sequence into a slow trend and a fast residual with the **same mid-cell DCT
basis the arch uses** (consistent-by-construction; no arch/OU variance double-counting). At fit time
`synthetic/fitting/register.py:register_moments_from_statistics` reduces the sums to moments and
`fit_register_config` maps them to a config (`θ = 1−ρ`, `σ = std·√(2θ−θ²)`, arch amplitude from
`band_limited_random`'s closed-form variance). Fitted overrides persist in a `FittedGeneratorConfig` artifact
and load into the **generation** path (`load_synthetic_inputs` → `build_segment_generator`); the calibration
sweep stays on the default (neutral), per the deferred coupling. Same `config + overrides` keying exists for
the accent field (#11).

*Design-vs-code refinement:* `generator.md` §3/§9 describes the register fit as a Welch-averaged PSD of
length-normalised trajectories; the implementation realises the same intent (moment-match arch + OU per
`(scale_type, hand)`) via the **mid-cell DCT trend/residual partition**, which is consistent-by-construction
with the arch sampler.

## 11. The empirical chord decode→generate loop — `D5` — CLOSED

Register and accent moment-matching are DB-backed (see "Moment-matching done" below); the empirical chord
loop — the last open piece of `D5` — is now closed too (see "Empirical chord loop done" below).

**Design (§5.1, §9).** The chord transition matrix and the chord-conditioned figure distribution
$p(\text{figure} \mid C)$ that drives the harmonic-fit score are fit by Viterbi-decoding the training
corpus; OU/arch/accent parameters are moment-matched against the corpus.

**Interim done.** Generation now uses `functional_transition_model` (`processes/chord_track.py`) — a
hand-authored functional diatonic prior keyed on root-degree function (tonic/predominant/dominant) with
V→I / vii°→I cadential weighting and a tonic-favoured initial distribution, blended with uniform by a
`strength` knob (`strength=0` reduces exactly to `uniform_transition_model`). This replaces the
uniform-random chord walk that was the dominant cause of incoherent harmony, *without* needing the corpus.
Calibration stays on `uniform_transition_model` (deferred coupling). `ViterbiChordDecoder` is still
unconsumed, and `harmonic_fit` still uses the chord pitch-class set, not $p(\text{figure} \mid C)$.

**Moment-matching done (register + accent), DB-backed.** Both global processes are now fit from persisted
corpus sufficient statistics (no on-the-fly recomputation). Register: see #10. Accent: the `config + overrides`
keying (`processes/accent.py`) is fed by a **per-within-bar onset-occupancy profile** stored during the
corpus pass as two rhythm kinds — `onset_position` (binary occupancy per bar×cell, per grid denominator) and
`bar_total` (the per-`(scale, time_signature, hand)` bar count). This closes the old "denominator gap":
`bar_total` *is* the slot denominator the strong/weak counts lacked. `synthetic/fitting/accent.py` pools the
occupancy by metrical **indispensability** (`gcd(k,M)/M`) across time signatures and fits a **3-parameter
weighted regression** of `logit(occupancy_rate)` on `indispensability^exponent` → `baseline_logit`,
`metric_gain`, and a **fitted `metric_exponent`** (searched over a small candidate set). The accent
**envelope** parameters remain pass-through (fitting deferred — see the deferred list).

*Design-vs-code refinement:* `generator.md` §9 describes the accent fit as fitting baseline/gain/exponent to
"the measured strong/weak onset ratio and overall density"; the implementation uses the richer **per-position
occupancy profile** instead (the strong/weak ratio is a two-bucket special case of it), which is what makes
the exponent identifiable.

**Empirical chord loop done (DB-backed, baked).** The decoder (relocated to the neutral
`musak_model/harmony/` package so the n-grams figure pass may consume it) now runs over every segment in the
figure-profile corpus pass. Two additive, mergeable count tables are persisted in a new `n_grams/profile/chord/`
sub-store: **chord transitions** keyed by `(scale_type, source_chord, destination_chord)` with the initial
chord folded in under a sentinel source, and **figure-by-chord** keyed by `(scale_type, hand, n, chord, figure)`
(the chord covering each figure's first onset). `ChordDecoderConfig` + the chord vocabulary enter
`figure_state_key`, so changing the decode forces a rebuild. The decode is confined to the durable reference
pass; the transient train/validation split pass stays decode-free.

`synthetic/fitting/chord.py` then fits, and `fit_generator.py` **bakes the results into `fitted_generator.json`**
(no on-the-fly recomputation at generation): per-scale-type empirical `ChordTransitionModel`s
(Dirichlet-smoothed toward the functional prior with `prior_count`; an unobserved source backs off exactly to
the prior) and a `FigureByChordModel` of $\log p(\text{figure} \mid C)$. The bake uses string chord keys
(`Chord.model_dump_json`) because pydantic does not round-trip model-keyed dicts. Generation consumes both: a
`chord_model: "uniform" | "functional" | "empirical"` selector picks the transition matrix (empirical falls
back to functional when a scale type was never fitted; calibration stays uniform), and a new
`lambda_chord_figure` tilt term adds $\log p(\text{figure} \mid C)$ to the substitution score — backing off
to `0` when the table or pair is unseen (an unobserved figure floors to the least-likely observed one, never
rewarded), so at `lambda_chord_figure = 0` output is byte-identical to before. The metrical chord-tone
`harmonic_fit` (#4) is unchanged; the new term **sharpens** it rather than replacing it.

## 12. λ-tilt selection is manual — `D6` — CLOSED

**Design (§9).** Run the sweep, score TV distance, and choose the largest tilt that keeps the mean TV
distance below a target threshold (a sensible initial choice is $0.1$).

**Code before.** `run_sweep` computed TV distance for every point on the product grid and wrote a CSV; no
selection rule was coded, so the choice was left to manual inspection.

**Resolution.** `calibration/selection.py:select_tilts` scans each direction independently with the other
two held at their baseline (0, or the grid minimum) and picks the largest λ whose mean TV distance is ≤
`CalibrationConfig.target_total_variation_distance` (default `0.1`, also in `calibration.yml`); when no
point on an axis clears the threshold it falls back to the lowest-TV point and reports
`threshold_met=False`. `calibrate` writes the chosen `SubstitutionConfig` to `<output>_selected.json` beside
the sweep CSV. Per-direction (not joint) selection was chosen to match the design's "independent dial"
framing.

## 13. Duplicate commonness-bias implementations — `D10` — CLOSED

**Code before.** `FigureVocabulary.sample` (`figures.py`) tilted the empirical frequencies by `count^β` but
was **unused** on the generation path; the live path uses `tilted_log_probabilities`
(`substitution/sampling.py`) with the equivalent `β · log(count)`.

**Resolution.** `FigureVocabulary.sample` and its `_entry_weight` / `_weighted_choice` helpers (test-only)
were removed along with their two tests. `tilted_log_probabilities` is now the single commonness-bias
implementation.

---

## Note on the greedy fill

`_emit_hand_bar` fills a bar left to right with no lookahead and rests the trailing gap when no sampled
figure fits the remaining time (flagged in its docstring). This is adequate for v1 but is a quality, not a
correctness, limitation; it is independent of the gaps above.
