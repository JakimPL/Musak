# Gaps between `docs/generator.md` and the current implementation

`docs/generator.md` is the design document. This file is the running list of places where the code does
not yet match it — gaps to be closed. Each entry names what the design says, what the code does today, and
what closing the gap requires.

## 1. Accent-field output is not consumed by substitution

**Design (§4, §6).** The per-cell probability $p_i(t)$ produced by the LGCP plays two roles: it gates
whether the cell becomes an onset, and it is the accent value surviving onsets carry forward into the
substitution step as the third tilt $\lambda_{\text{accent}} \, A(f, \lambda_i(k))$ of the I-projection.

**Code today.** `AccentFieldSampler` in `musak_model/synthetic/processes/accent.py` fully computes the
LGCP and emits `AccentCell.weight`. `SegmentGenerator.sample` in
`musak_model/synthetic/substitution/generator.py` never instantiates the sampler, so the accent value
reaches no consumer; the tilted log-probabilities only see $\lambda_{\text{curve}} S$ and
$\lambda_{\text{harm}} H$.

**To close the gap.** Instantiate `AccentFieldSampler` in `SegmentGenerator`, plumb each anchor cell's
weight through to the substitution step, and add the $A$ score — figure internal accent shape against the
envelope — alongside the existing slope and harmony scores.

## 2. Register-curve home-offset $\mu_i$ is dropped

**Design (§3).** The trajectory is $P_i(k) = \mathrm{round}(\mu_i + P_i^{\text{arch}}(k) + r_k)$ with
$\mu_i = o_i \cdot s$ — five-octave home for the right hand, three-octave home for the left — so the curve
anchors each hand on its own register.

**Code today.** `RegisterCurveSampler.sample` in `musak_model/synthetic/processes/pitch.py` returns
`np.rint(arch + residual)`; the `scale_type` and `hand` arguments are accepted and immediately discarded
(`_ = scale_type, hand`). The trajectory is zero-centred, so both hands sample around the same diatonic
position.

**To close the gap.** Compute `mu_i = HAND_HOME_OCTAVES[hand] * scale_size_for_type(scale_type)` inside
the sampler and add it into the lattice-quantised sum; cover with a test that the long-run mean of the
trajectory tracks $\mu_i$ per hand.

## 3. Decoder windowing can append a length-0 tail window

**Design (§5.1).** The chord grid is laid down from the downbeat at the configured resolution and
truncated at the next barline; every emitted window is a contiguous slice of the bar containing the
sounding pitch classes that fell inside it.

**Code today.** `musak_model/synthetic/harmony/decoding/windows.py` clips
`window_end = min(window_start + window_value, next_bar_boundary, total_duration)` and appends a window
every iteration. When the resolution is finer than the bar fragment remaining after truncation,
`window_end == window_start` can hold and a length-0 window enters the Viterbi pass with empty
`pitch_class_weights`. Such a window scores all candidates identically and biases the backtrack toward
whichever candidate is chosen first by ties.

**To close the gap.** Guard the append with `if window_end > window_start` in `windows.py`; no other
change is needed.
