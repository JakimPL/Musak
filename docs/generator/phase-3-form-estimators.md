# Phase 3 — Form estimators: learning the `FormPrior` from the corpus

> Concrete design for Phase 3 batch 3 of the coherence redesign ([coherence.md](coherence.md) §11/§16/§17/§19).
> Batches 1–2 (the generative form layer + per-phrase harmony) are in place: `FormSampler(prior).sample(...) →
> FormTree` and `SurfaceRenderer.render(form=...)` already reproduce periods. This batch supplies the **learned**
> half — the offline estimators that read decoded harmony + figures and produce, per `ScaleType`, a `FormPrior`
> (phrase-length distribution, segment-length distribution, `P(ClosingPattern | is_final)`, and repeat/variation
> rates).

The research-heart of the generator is here: the **cadence detector** (measured, not enumerated — coherence §19)
and the **repetition-similarity** estimator.

Locked decisions:
- **Persistence:** mirror the figure pipeline — a resumable **SQLite work-store** (`form.sqlite3`, batched,
  `completed_batches`) → export to **parquet** count tables → read by the fit step. Same idiom as
  `n_grams/profile/streaming/` + `n_grams/profile/chord/`.
- **Repetition thresholds:** **data-calibrated** from a persisted corpus similarity histogram (Otsu valley), with
  a YAML fallback when a scale type's corpus is too sparse to be bimodal.
- **Scope:** short, single-key exercises. The **live corpus run is deferred** (a heavy pass needing the actual
  encoded corpus); this batch delivers the machinery + fixture tests. Running it is a `make`-style task, exactly
  like `extract_figures.py` → `fit_generator.py`.

**Dependency direction** (per [guidelines](../guidelines.md)): the estimator imports from `n_grams` (figure
extraction, Viterbi decoder) and `synthetic.structure` (`FormPrior`, `ClosingPattern`), so it **lives under
`synthetic/`**, never under `n_grams/profile/` (which may not depend on `synthetic`). It therefore gets its own
`form.sqlite3` rather than piggy-backing the figure `FigureWorkStore`.

---

## What the estimators consume (per corpus exercise)

The corpus pass reads the **same encoded JSONL** the figure pipeline reads (`EncodedExercise` →
`sample.to_segment(...)` → `Segment`), so form statistics are computed over exactly the material figures are.

### Dataset curation (prerequisite)

Form is a **whole-piece** property: cadence spacing and parallelism only mean something when a sample is an entire
short piece, not an arbitrary mid-phrase window. So the corpus for the form pass must be **curated and limited to
relatively short, complete pieces** and encoded with `SegmentationMode.WHOLE_FILE` (one sample = one whole piece),
filtered to a bar-count band matching the generator's target (≈4–16 bars). The cadence detector is nonetheless
robust to window edges — interior cadences are found regardless of where a window starts; only `is_final` needs a
true end, which it reads from the segment's last bar — but the phrase-length and repetition statistics degrade on
arbitrarily-windowed long pieces. Curation is the cheapest, highest-leverage way to keep the learned `FormPrior`
honest.

### Per-`Segment` derivation (`AnalyzedPiece`)

From existing helpers:
- **Decoded chords** via `ViterbiChordDecoder(config).decode(segment, duration_vocabulary, vocabulary)` →
  `tuple[ChordWindow, ...]` (`harmony/decoding/decoder.py`). Consecutive equal `Chord`s are merged into
  **harmonic slots** (the chord progression).
- **Per-bar figure counters** for repetition: `extract_hand_onset_runs(tokens, …)` (`n_grams/figure/parser.py`),
  then `count_figure_ngrams([run], min_n, max_n, scale_size=scale_size_for_type(scale_type))` per run, bucketed
  into `bar_figures[bar]` (combined hands, all n). Runs already break at bar boundaries, so every n-gram belongs to
  one bar.

`synthetic/fitting/form/analysis.py`:
```
HarmonicSlot:    start: Fraction; end: Fraction; bar_index: int; chord: Chord
                 function: HarmonicFunction | None; metrical_weight: float
                 tonic_triad_overlap: float; dwell: float          # dwell = (end-start)/bar_duration
AnalyzedPiece:   scale_type: ScaleType; bar_count: int; bar_duration: Fraction
                 slots: tuple[HarmonicSlot, ...]; bar_figures: tuple[Counter[FigureNGram], ...]

analyze_segment(segment, *, chord_decoder, chord_vocabulary, duration_vocabulary) -> AnalyzedPiece | None
```
Per slot:
- `function = HARMONIC_FUNCTION_BY_DEGREE.get(chord.root_degree)` (`musak_shared/elements.py`; `None` for
  chromatic roots not in the diatonic map).
- `metrical_weight`: reuse `indispensability_per_position(cells_per_bar)[cell_index]`
  (`processes/accent.py` = `gcd(k, M)/M`), with `cell_index = round((start mod bar_duration)/window_duration)` and
  `cells_per_bar` from the decoder resolution. This is the corpus-side "metrical strength" (= tree rank on regular
  meters, coherence §13.1).
- `tonic_triad_overlap = |PC(chord) ∩ PC(tonic)| / |PC(tonic)|` using
  `chord_pitch_class_set(chord, scale_type, vocabulary)` (`harmony/expansion.py`) and `natural_triad(scale_type, 1)`
  (`harmony/diatonic.py`). Never hardcode 7 — use `scale_size_for_type`.

---

## Algorithm 1 — Cadence detection (measured, non-circular) — `cadence.py`

A cadence is an **arrival** at a metrically strong position with harmonic relaxation and a rhythmic stop (the
invariant confirmed by the cadence-detection literature: a dominant in a metrically strong position
resolving/relaxing). We **score** independently-measurable arrival cues and threshold — we do **not** pattern-match
named cadence types, and we do not classify into PAC/IAC/HC/DC. The output is a **measured `ClosingPattern`**
(functional suffix), so `P(ClosingPattern | is_final)` is whatever the corpus yields (coherence §19 inductive-bias
policy: the T/S/D division is an accepted bias for sight-reading; the *categorization into cadence names is not*
imposed).

For each slot `j` as a candidate arrival:
```
metrical_strength(j) = slot[j].metrical_weight
authentic_arrival(j) = tonic_triad_overlap(j) · 1[func(j-1)=DOMINANT] · 1[func(j)=TONIC]
half_arrival(j)      = 1[func(j)=DOMINANT] · min(1, slot[j].dwell)          # dwell on the dominant
harmonic_arrival(j)  = max(authentic_arrival(j), half_arrival(j))
rhythmic_stop(j)     = min(1, slot[j].dwell)                                 # long held arrival chord
bar_aligned(j)       = 1[slot[j].end is an integer multiple of bar_duration]

boundary_score(j) = w_metrical·metrical_strength(j) + w_harmonic·harmonic_arrival(j)
                  + w_stop·rhythmic_stop(j)         + w_bar·bar_aligned(j)
```
Weights `w_*` ∈ `FormFittingConfig` (YAML). `j` is a **cadence** iff `boundary_score(j) ≥ cadence_threshold` **and**
it is a local maximum within `±minimum_cadence_separation_slots` (non-maximum suppression — prevents double
counting). The **piece-final slot is always a cadence**.

For each detected cadence at slot `j`:
- **`ClosingPattern`**: walk left from `j` collecting `func(...)`, deduplicating consecutive equals, stopping after
  the first PREDOMINANT that precedes the DOMINANT or after `maximum_closing_slots`; e.g.
  `(PREDOMINANT, DOMINANT, TONIC)`, `(DOMINANT, TONIC)`, `(PREDOMINANT, DOMINANT)`, `(DOMINANT,)`.
  `terminal_function = func(j)`. Slots with `function is None` break the suffix.
- **`is_final`** = `j` is the final cadence of the piece.
- **phrase length** = `end_bar(j) − end_bar(previous cadence)` (first phrase measured from bar 0).

`detect_cadences(piece, *, config) -> tuple[Cadence, ...]`,
`Cadence: arrival_slot_index:int; end_bar:int; closing:ClosingPattern; is_final:bool; boundary_score:float`.

> The cadence weights/threshold are the **one hand-tuned analysis component** — justified (coherence §7 L1) because
> the corpus has no ground-truth cadence labels, so we measure with an interpretable heuristic rather than train a
> classifier we cannot supervise. They live in YAML and can be calibrated later against the structural metrics.

---

## Algorithm 2 — Repetition similarity → repeat/variation rates — `repetition.py`

`FormSampler` only needs the **marginal** repeat/variation rates (it draws fresh/same/variant per segment via
Bernoullis) — not the full canonical string — so we estimate those rates directly, which also resolves the two-pass
threshold dependency cleanly.

Per piece, for each candidate segment length `g ∈ segment_length_candidates` (config, e.g. `[1, 2, 4]`):
- partition bars into `⌈bar_count/g⌉` segments; each segment's figure counter = `Σ bar_figures[start:start+g]`.
- pairwise `similarity(a, b) = 1 − total_variation_distance(counter_a, counter_b)`
  (`n_grams/profile/metrics/stats.py`; `FigureNGram` is frozen/hashable so `Counter[FigureNGram]` works). Degrees
  in `FigureNGram` are figure-root-relative → transposition-invariant by construction (coherence §11.3.2).

Choose **`g*`** = the candidate maximizing the piece's mean per-segment **best-earlier-match** similarity
(threshold-free; ties → larger `g`). Record `segment_length = g*`.

The worker accumulates (no labeling yet — thresholds aren't known mid-pass):
- **similarity histogram**: every pairwise `similarity` at `g*`, bucketed → used at fit time for the Otsu valley.
- **best-match histogram**: each segment's max similarity to an earlier segment, bucketed → integrated at fit time
  (with calibrated thresholds) into `repeat_probability` / `variation_probability`.

`analyze_repetition(piece, *, config) -> RepetitionAnalysis`
(`segment_length:int; pairwise_similarities:tuple[float,...]; best_match_similarities:tuple[float,...]`).

---

## Persistence — resumable SQLite work-store (mirror `n_grams/profile/streaming/`)

`synthetic/fitting/form/store.py` — `FormWorkStore` + `FormWorkTables`, in the established idiom
(`create_engine("sqlite:///…")`, WAL/synchronous PRAGMAs, `MetaData.create_all`, `state_key` validation,
`completed_batches`, additive-upsert via `insert(...).on_conflict_do_update`, context-manager lifecycle). DB file
`form.sqlite3` under the figure root. Five count tables (grain = primary key, per the row-grain convention):

| table | primary key (grain) | value |
|---|---|---|
| `phrase_length_counts`   | (scale_type, phrase_length_bars)   | count |
| `segment_length_counts`  | (scale_type, segment_length_bars)  | count |
| `closing_counts`         | (scale_type, is_final, functions)  | count   (`functions` = `>`-joined `HarmonicFunction` values) |
| `similarity_histogram`   | (scale_type, bucket)               | count |
| `best_match_histogram`   | (scale_type, bucket)               | count |

plus `metadata` (state_key) and `completed_batches` (resume), exactly as `FigureWorkTables`.

- `worker.py` — `process_form_batch_task(task) -> FormBatchResult`: for each `EncodedExercise` line → `Segment` →
  `analyze_segment` → `detect_cadences` + `analyze_repetition` → increment the five Counters.
- `executor.py` — `form_batch_tasks(encoded_jsonl_path, …, completed_batches)` and
  `process_missing_form_batches(store, …)` (serial + `ProcessPoolExecutor` parallel), reusing
  `processing.workers.process_pool_context` and `processing.progress.progress`.
- `orchestration.py` — `extract_form_statistics(...)`: open store, `process_missing_form_batches`, then **export**
  to parquet; `form_state_key(...)` over the decoder + config (mirrors `figure_state_key`).
- `io.py` + parquet column/schema constants (mirror `n_grams/profile/chord/io.py` with `write_table`/`read_table`
  from `musak_shared.tables`): `form/<table>.parquet` under `figure_root/form/`; readers return the same Counters.
  `form_artifact_paths_for_figure_root(figure_root)` mirrors `chord_artifact_paths_for_figure_root`.

---

## The fit step — `synthetic/fitting/form/fit.py`

`FormFittingConfig` (frozen pydantic, `.load(path=FORM_FITTING_CONFIG_PATH)`):
- cadence detector: `metrical_weight`, `harmonic_arrival_weight`, `rhythmic_stop_weight`, `bar_alignment_weight`,
  `cadence_threshold`, `minimum_cadence_separation_slots`, `maximum_closing_slots`.
- repetition: `segment_length_candidates: tuple[int, …]`, `similarity_bucket_count`, `same_similarity_threshold`,
  `variation_similarity_threshold` (the **fallback** values).
- smoothing/back-off: `smoothing_pseudo_count`, `minimum_observation_count`, and a nested **`fallback_prior`** (a
  full `FormPrior` literal — defaults live in YAML, not code).

`fit_form_priors(stats, *, config) -> dict[ScaleType, FormPrior]` per scale type:
1. **Calibrate thresholds** from `similarity_histogram`: `variation_threshold` via **Otsu** over `[0, 1]`;
   `same_threshold` via Otsu over the sub-histogram `≥ variation_threshold` (recursive, parameter-light). If a
   region has `< minimum_observation_count` mass → fall back to the config thresholds.
2. **repeat/variation rates** by integrating `best_match_histogram`:
   `repeat_probability = mass(≥ variation_threshold) / total`;
   `variation_probability = mass([variation_threshold, same_threshold)) / mass(≥ variation_threshold)`
   (Dirichlet-smoothed with `smoothing_pseudo_count` toward the fallback's rates).
3. **phrase_lengths / segment_lengths** → `WeightedSpan`s from the count tables, weights =
   `count + smoothing_pseudo_count · fallback_probability(bars)`.
4. **closings** → `ClosingChoice`s from `closing_counts` (parse `functions` back to `HarmonicFunction` tuples), same
   smoothing; **guarantee** ≥1 `is_final=True` and ≥1 `is_final=False` choice (inject from `fallback_prior` if the
   corpus lacks one) — `FormSampler._sample_closing` raises otherwise.
5. If a scale type's total observations `< minimum_observation_count`, return the `fallback_prior` outright.

---

## Wiring into the fitted artifact & generator

- `synthetic/fitting/artifacts.py`: `FittedGeneratorConfig` gains `form_priors: dict[ScaleType, FormPrior] = {}`
  (FormPrior is already frozen pydantic → serializes natively into `fitted_generator.json`, exactly like
  `chord_transitions`) and an accessor `form_prior(self, scale_type) -> FormPrior | None`.
- `synthetic/fitting/fit.py` (`fit_generator_config`): read the form parquet artifacts under `figure_root`
  (graceful absence → `{}`, mirroring `_fit_chord_transitions_from_store`) and call `fit_form_priors(...)`; thread a
  `form_fitting: FormFittingConfig` parameter through (like `chord_fit`).
- `scripts/extract_form_statistics.py` (new, thin — mirrors `scripts/extract_figures.py`): resolve the encoded dir,
  build the decoder spec, call `extract_form_statistics(...)`.
- `scripts/fit_generator.py`: load `FormFittingConfig`, pass it through, log `len(fitted.form_priors)`.
- **Generator consumption (minimal, correct for this phase):** the new `SurfaceRenderer` is not yet assembled in
  production (`builder.py` still wires the *old* `SegmentGenerator`; full assembly is Phase 5, coherence §17). So
  the Phase-3 wire is just: `FittedGeneratorConfig.form_prior(scale_type)` (fallback to the YAML default prior) →
  `FormSampler(prior).sample(bar_count, rng)` → `SurfaceRenderer.render(form=…)`, demonstrated by a test;
  production `SurfaceRenderer` assembly stays Phase 5.

Config & paths: `musak_model/configs/generation/form_fitting.yml` (all of the above, uncommented);
`musak_model/paths.py` → `FORM_FITTING_CONFIG_PATH = CONFIGS_DIRECTORY / "generation" / "form_fitting.yml"`.

---

## Files (new unless noted)

```
musak_model/synthetic/fitting/form/__init__.py
musak_model/synthetic/fitting/form/analysis.py      # AnalyzedPiece, HarmonicSlot, analyze_segment
musak_model/synthetic/fitting/form/cadence.py       # detect_cadences, Cadence, boundary scoring + NMS
musak_model/synthetic/fitting/form/repetition.py    # analyze_repetition, similarity, g* selection
musak_model/synthetic/fitting/form/store.py         # FormWorkStore + FormWorkTables (sqlite, resumable)
musak_model/synthetic/fitting/form/worker.py        # process_form_batch_task, FormBatchTask/Result
musak_model/synthetic/fitting/form/executor.py      # form_batch_tasks + process_missing_form_batches
musak_model/synthetic/fitting/form/orchestration.py # extract_form_statistics, state_key
musak_model/synthetic/fitting/form/io.py            # parquet write/read of the 5 tables + path helper
musak_model/synthetic/fitting/form/fit.py           # FormFittingConfig, fit_form_priors, Otsu + smoothing
musak_model/configs/generation/form_fitting.yml
scripts/extract_form_statistics.py
musak_model/synthetic/fitting/artifacts.py          # (edit) + form_priors + accessor
musak_model/synthetic/fitting/fit.py                # (edit) read form artifacts, call fit_form_priors
musak_model/paths.py                                # (edit) + FORM_FITTING_CONFIG_PATH
scripts/fit_generator.py                            # (edit) load FormFittingConfig, log count
```

Conventions: frozen pydantic for serialized objects, frozen dataclasses for internal state, verbose names (no
abbreviations), `Final` constants, reuse existing helpers (decoder, `extract_hand_onset_runs`,
`count_figure_ngrams`, `total_variation_distance`, `indispensability_per_position`, `chord_pitch_class_set`,
`natural_triad`, `degrees_for_function`, `scale_size_for_type`, `musak_shared.tables`), no hardcoded scale sizes,
minimal documentation. Run tooling via `uv run`.

---

## Tests (fixture-based; mirror package layout; the live corpus run stays deferred)

- `test_analysis.py` — a hand-built `Segment` → `analyze_segment` yields slots with the right
  function/metrical_weight/tonic_overlap; chromatic root → `function is None`.
- `test_cadence.py` (the heart) — synthetic `AnalyzedPiece`s: V→I on a strong beat is detected with terminal TONIC;
  a strong dwell on V yields a DOMINANT-terminal closing; NMS suppresses a weaker adjacent peak; the final slot is
  always a cadence; phrase lengths come out as the inter-cadence bar spans.
- `test_repetition.py` — two identical `g`-segments → similarity ≈ 1; `g*` selection picks the periodic length;
  best-match histogram reflects the repeat.
- `test_fit.py` — bimodal `similarity_histogram` → Otsu valley near the gap; smoothing blends toward
  `fallback_prior`; sparse scale type → returns the fallback; output passes `FormPrior`/`FormSampler` validation;
  round-trips through `FittedGeneratorConfig` JSON; closings always include an `is_final` true and false choice.
- `test_store.py` — `FormWorkStore` open/commit/reopen: additive upserts accumulate; `completed_batches` skips
  re-processing (resume).
- `test_wiring.py` — `FittedGeneratorConfig.form_prior(scale_type)` → `FormSampler` → `SurfaceRenderer.render`
  produces a constraint-valid segment.

## Verification (end-to-end, when run on the live curated corpus — deferred)
1. `uv run python -m scripts.extract_form_statistics --data-dir … --grid-denominator …` → `form.sqlite3` +
   `form/*.parquet`.
2. `uv run python -m scripts.fit_generator --data-dir … --grid-denominator …` → `fitted_generator.json` now carries
   `form_priors`; log shows the per-scale-type prior count.
3. Inspect sampled `FormTree`s vs corpus: cadence-spacing (phrase-length) and `P(closing | is_final)` match the
   corpus histograms; the 8-bar period reproduces (coherence §15).
4. Gate: `uv run pytest tests/musak_model/synthetic`, `uv run mypy`, `uv run pylint`, `uv run isort --check`.
