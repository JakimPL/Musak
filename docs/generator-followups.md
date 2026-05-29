# Gaps between `docs/generator.md` and the current implementation

`docs/generator.md` is the design document. This file is the running list of places where the code does
not yet match it — gaps to be closed. Each entry names what the design says, what the code does today, and
what closing the gap requires.

The earlier gaps are closed: the accent field is wired into substitution, and the length-0 decoder window
is guarded. (The register home-offset $\mu_i$ was never actually a gap — `octave_offset` is home-relative
and $\mu_i$ is applied at token-to-MIDI conversion; see `docs/generator.md` §3.) Gaps 1 and 3 are now
closed too: `SegmentGenerator.generate` takes a `grid_count_per_bar` and drives emission from a sub-bar
onset grid. The remaining gap (#2, sync) still resolves hand interaction below the grid the design assumes.

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

## 2. Sync coupling is not implemented

**Design (§7).** Hand interaction has three couplings — co-activity, **sync** ($h_s$: the probability that
both hands' attacks coincide on the grid), and shared harmony. Co-activity and shared harmony exist;
sync does not.

**Code today.** `HandCouplingSampler` models only co-activity (the Gaussian-copula gate). There is no
sync parameter and no mechanism aligning the two hands' attacks within a bar.

**To close the gap.** Sync only carries information once a bar holds more than one onset per hand, so it
depends on gap 1. Add $h_s$ to `HandCouplingConfig` and bias the two hands' onset grids toward coincident
attacks.

## 3. Register and accent are shared across all figures in a bar — CLOSED

**Design (§3, §6).** The register curve yields an integer diatonic position *per onset step*, and the
accent value entering the substitution tilt is the envelope *at the current cell*.

**Resolution.** The register curve is now sampled at grid resolution (`length =
bar_count * grid_count_per_bar`) and each figure reads its anchor from the firing cell, its slope target
from that cell to the next, and its envelope value from the firing cell's accent weight, so sub-bar
register motion and per-cell accent shaping are modelled.

## Note on the greedy fill

`_emit_hand_bar` fills a bar left to right with no lookahead and rests the trailing gap when no sampled
figure fits the remaining time (flagged in its docstring). This is adequate for v1 but is a quality, not a
correctness, limitation; it is independent of the gaps above.
