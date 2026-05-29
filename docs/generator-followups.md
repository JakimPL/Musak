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

---

## 1. Activity gating is bar-resolution, not grid-cell-resolution — CLOSED

**Design (§4, §7).** The accent field is a marked point process on the bar grid; each grid cell is
independently an onset or a rest, and the hand-coupling gate acts per cell, so a hand can fall silent for
part of a bar.

**Resolution.** `SegmentGenerator.generate` now takes `grid_count_per_bar` and samples the accent field
(`AccentFieldSampler.sample`, with onset marks) and the hand-coupling gates at `cell_count =
bar_count * grid_count_per_bar`. A cell fires iff its accent is an onset *and* the per-cell coupling gate
is active for the hand; the cursor walks each bar, resting unfired stretches and starting a figure at each
fired cell, so a hand can fall silent mid-bar. Figure durations are unchanged — a figure may still span
many cells; the grid only fixes onset times.

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

## 4. Harmonic-fit tilt ignores metrical position — `D1`

**Design (§6).** The harmonic-fit score is $H(f, C, m)$: chord tones are rewarded on strong beats and
non-chord tones are permitted — indeed favoured — on weak beats *between* chord tones, at a weight that
depends on the figure's metrical position $m$ in the bar.

**Code today.** `harm_fit` (`substitution/scoring.py:21`) returns the plain chord-tone fraction over all of
the figure's note instances; it takes no metrical position and applies no strong/weak gradient. The cell's
metrical position is available at the call site (`fired_cell_index % grid_count_per_bar`, and the accent
field's indispensability `gcd(k, M)/M`) but is not threaded in.

**To close the gap.** Pass the firing cell's metrical position (or its indispensability) into `harm_fit`,
and make the per-onset chord-tone reward scale with metrical strength so NCTs are tolerated/encouraged on
weak cells. Open design question: the score must account for *which onset of the figure lands on which
beat*, not just the figure as a whole — decide the functional form before coding.

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
(`generator.py:347`) — uniform over `[min_n, max_n]`. `FigureVocabulary.length_distribution()` exists but is
never used on the generation path.

**To close the gap.** Sample the length from `length_distribution()` (optionally per scale/hand), or fold
length into a single tilted choice over a combined candidate pool. **Check before changing:** calibration
scores TV distance per `(scale, hand, n)` group independently, so uniform length keeps every `n` populated
for the metric — confirm against `figure_distribution_metrics` whether uniform sampling is intentional.

## 8. Monorhythmic filtering is unused; chord/polyrhythmic figures are admitted — `D8`

**Design (§10).** Figures own local contour and rhythm; emission supports chords (via
`JoinWithPreviousToken`). The functional-bass / block-chord texture is an *optional* mode, not the default.

**Code today.** `monorhythmic_entries` / `is_monorhythmic` (`substitution/sampling.py`, `scoring.py`) exist
but are not called by the generator; `_figure_entries_by_group` (`generator.py:219`) admits *all* figures in
each `(hand, n)` group, including `chords_only` and polyrhythmic ones.

**To close the gap.** Make the texture explicit rather than incidental: either (a) keep all figures by
design and remove/annotate the now-dead `monorhythmic_entries` helper (confirm it is test-only), or (b) add
a texture-mode knob (monophonic melodic lines vs. chordal/Alberti) and filter the candidate pool
accordingly.

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

## 10. Register curve is hand- and scale-agnostic — `D4`

**Design (§3, §9).** The OU parameters $(\theta_i, \sigma_i)$ and arch amplitudes $\{A_j\}$ are per
`(scale_type, hand)` — roughly four numbers per group — fit from the empirical register spread, lag-1
autocorrelation, and low-frequency PSD.

**Code today.** `RegisterCurveSampler.sample` explicitly discards `scale_type` and `hand`
(`processes/pitch.py:49`, `_ = scale_type, hand`); both hands draw i.i.d. from one shared
`RegisterCurveConfig`. The only L/R difference is the home octave added at `note_token_to_midi_pitch`.

**To close the gap.** Key the register config by `(scale_type, hand)` (a mapping, or per-group config
objects) and wire the moment-matching fit (see #11). Even before fit data exists, allow per-hand config so
the two hands' spreads can differ.

## 11. Process parameters are hand-set; the decode→generate fitting loop is open — `D5`

**Design (§5.1, §9).** The chord transition matrix and the chord-conditioned figure distribution
$p(\text{figure} \mid C)$ that drives the harmonic-fit score are fit by Viterbi-decoding the training
corpus; OU/arch/accent parameters are moment-matched against the corpus.

**Code today.** `ViterbiChordDecoder` (`harmony/decoding/`) exists and is correct, but its output is never
consumed: `ChordTrackSampler` is always built from `uniform_transition_model`
(`calibration/assembly.py`, `notebooks/utils/synthetic.py`), and `harm_fit` reads a chord pitch-class set,
not $p(\text{figure} \mid C)$. Every process config is a static YAML / slider value. The decode→generate
calibration loop is an unconnected seam.

**To close the gap (largest item — likely several PRs).**
- Add a fitting pass that runs the decoder over the corpus, accumulates the empirical chord transition
  matrix and figure-by-chord co-occurrence counts, and persists them as artifacts.
- Build `ChordTransitionModel` from the empirical matrix; replace the uniform prior in assembly.
- Augment `harm_fit` to read the empirical $p(\text{figure} \mid C)$ conditional per design §6, rather than
  only the binary chord-tone test (interacts with #4).
- Add moment-matching for the register and accent parameters (interacts with #10).

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
