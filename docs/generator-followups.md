# Gaps between `docs/generator.md` and the current implementation

`docs/generator.md` is the design document. This file is the running list of places where the code does
not yet match it — gaps to be closed. Each entry names what the design says, what the code does today, and
what closing the gap requires.

The earlier gaps are closed: the accent field is wired into substitution, and the length-0 decoder window
is guarded. (The register home-offset $\mu_i$ was never actually a gap — `octave_offset` is home-relative
and $\mu_i$ is applied at token-to-MIDI conversion; see `docs/generator.md` §3.) The remaining gaps below
share a single root cause — the generator still resolves register, accent, and hand activity at **bar**
resolution rather than at the **grid-cell** resolution the design assumes — and are best taken as one
coherent unit of work.

## 1. Activity gating is bar-resolution, not grid-cell-resolution

**Design (§4, §7).** The accent field is a marked point process on the bar grid; each grid cell is
independently an onset or a rest, and the hand-coupling gate acts per cell, so a hand can fall silent for
part of a bar.

**Code today.** `SegmentGenerator.generate` samples the accent envelope and the hand-coupling gates with
one cell per bar (`grid_count_per_bar=1`, `cell_count=bar_count`). A hand is therefore either active for a
whole bar or silent for a whole bar; there is no sub-bar gating. The samplers themselves already accept a
finer grid — only the caller fixes it to one cell per bar.

**To close the gap.** Drive the per-bar emission from a sub-bar onset grid: sample the accent field and
gates at the real grid resolution, and let fired cells start figures while unfired cells become rests.

## 2. Sync coupling is not implemented

**Design (§7).** Hand interaction has three couplings — co-activity, **sync** ($h_s$: the probability that
both hands' attacks coincide on the grid), and shared harmony. Co-activity and shared harmony exist;
sync does not.

**Code today.** `HandCouplingSampler` models only co-activity (the Gaussian-copula gate). There is no
sync parameter and no mechanism aligning the two hands' attacks within a bar.

**To close the gap.** Sync only carries information once a bar holds more than one onset per hand, so it
depends on gap 1. Add $h_s$ to `HandCouplingConfig` and bias the two hands' onset grids toward coincident
attacks.

## 3. Register and accent are shared across all figures in a bar

**Design (§3, §6).** The register curve yields an integer diatonic position *per onset step*, and the
accent value entering the substitution tilt is the envelope *at the current cell*.

**Code today.** The register curve is sampled per bar (`length=bar_count`) and every figure placed in a
bar shares that bar's single anchor, slope target, and accent value (see the `_emit_hand_bar` docstring).
Sub-bar register motion and per-cell accent shaping are therefore not modelled.

**To close the gap.** Sample register and accent at the onset-grid resolution (gap 1) and read the anchor,
slope, and envelope per placement rather than per bar.

## Note on the greedy fill

`_emit_hand_bar` fills a bar left to right with no lookahead and rests the trailing gap when no sampled
figure fits the remaining time (flagged in its docstring). This is adequate for v1 but is a quality, not a
correctness, limitation; it is independent of the gaps above.
