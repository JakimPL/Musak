# Symbolic Music Generation Literature Review

This document reviews symbolic music generation work that is directly relevant to Musak's sight-reading model. It
intentionally focuses on symbolic piano- or score-like generation, controllability, representation, long-range
structure, and evaluation. Audio-only and text-to-audio systems are out of scope unless their control ideas transfer
cleanly to symbolic generation. The review is current as of 2026-05-30.

## Executive Summary

Current symbolic-music work suggests that a plain next-token Transformer is rarely enough. The strongest recurring
patterns are:

1. **Expose musical coordinates explicitly.** Bar, beat position, duration, pitch, instrument, chord, and control
   attributes are often represented as separate fields or events instead of one undifferentiated token stream.
2. **Use hierarchy.** Strong systems separate local events from bar-, phrase-, theme-, or song-level structure.
3. **Condition generation on musical plans.** Chords, rhythmic density, style attributes, theme material, or future
   control events are treated as first-class conditions.
4. **Use constraints and references as inductive bias.** Many systems improve musicality by injecting domain knowledge:
   relative attention, bar-level masking, similarity-selected bars, high-level descriptors, or explicit skeletons.
5. **Evaluate beyond loss.** Likelihood is not sufficient. Objective statistics, control accuracy, structure metrics,
   and human listening remain necessary.

For Musak, this points toward a two-layer design: a low-entropy planner over musical structure and figure families,
followed by a reference-backed renderer that emits valid token sequences.

## Taxonomy

### Representation

Early Transformer-style symbolic systems often used MIDI-like event streams, but later work increasingly adds metrical
and typed structure. REMI adds bar and beat-position events to give the model rhythmic and harmonic context. Compound
Word Transformer groups several event fields into one compound step, reducing sequence length and exposing the fact
that pitch, duration, velocity, and position are different coordinates. MusicBERT's OctupleMIDI uses a similar
multi-field idea for symbolic pretraining and combines it with bar-level masking.

Musak's current token vocabulary is also structured internally, but this structure is flattened into a single token ID.
That makes the model relearn the cross-product of degree, accidental, octave, and duration from data. A factorized
head or compound-token objective would align Musak with the stronger representation trend without abandoning the
existing scale-degree semantics.

### Long-Range Structure

Music Transformer showed that relative attention helps model longer musical dependencies than earlier sequence models.
MusicVAE used hierarchy in the latent decoder to model long-term structure. Museformer goes further by making attention
bar-aware: tokens attend finely to structure-related bars and coarsely to summaries of other bars. Theme Transformer
adds explicit theme conditioning so a motif can reappear and develop rather than merely serve as a prompt.

The shared lesson is that long-range musical form is not just "more context." The model needs a structural path for
repetition, contrast, bar relationships, and phrase-level intent.

### Controllability

FIGARO uses high-level musical descriptions and domain knowledge as control codes. Anticipatory Music Transformer
conditions event generation on asynchronous control events, which is useful when controls are known ahead of the
notes but should apply at future musical times. MuseCoco separates text-to-attribute understanding from
attribute-to-music generation, using musical attributes as the controllable bridge.

Musak's user-facing goal is already control-oriented: scale, meter, difficulty, density, range, and exercise style.
The literature supports making these controls explicit generation objects rather than hoping a raw-token model learns
them from a single prefix embedding.

### Reference And Constraint Use

The most relevant non-neural lesson is that references can guide generation without becoming tokens. In Musak, figure
n-grams are too numerous and sparse to become a vocabulary, but they are well suited as:

- empirical priors over local idioms;
- transition statistics between coarse figure families;
- reference distributions for calibration;
- decoding-time scores or renderer emissions;
- evaluation baselines.

This matches the direction of the synthetic generator design in [generator.md](generator.md): low-order global
processes supply structure, and empirical figure templates supply local musical surface.

## Paper Notes

| Work | Main Idea | Musak Relevance |
| --- | --- | --- |
| Music Transformer, 2018 ([paper](https://research.google/pubs/music-transformer-generating-music-with-long-term-structure/)) | Relative attention for long-range symbolic music generation and accompaniment. | Supports adding music-aware attention or bar-relative structure instead of relying on absolute token position alone. |
| MusicVAE, 2018 ([paper](https://arxiv.org/abs/1803.05428)) | Hierarchical latent model for longer musical sequences. | Supports a planner/latent layer above token-level emission. |
| Pop Music Transformer / REMI, 2020 ([paper](https://arxiv.org/abs/2002.00212)) | Beat-based representation with bar, position, tempo, chord, and note events. | Supports explicit metrical and harmonic coordinates in the model input/output. |
| Compound Word Transformer, 2021 ([paper](https://cdn.aaai.org/ojs/16091/16091-13-19585-1-2-20210518.pdf)) | Groups multiple token fields into compound events for full-song generation. | Strong precedent for factorizing Musak note tokens into typed heads. |
| MusicBERT, 2021 ([paper](https://arxiv.org/abs/2106.05630)) | OctupleMIDI and bar-level masking for symbolic music pretraining. | Supports bar-level objectives and structured token fields for data-limited symbolic corpora. |
| Theme Transformer, 2021 ([paper](https://arxiv.org/abs/2111.04093)) | Theme-conditioned generation with mechanisms that encourage repeated thematic material. | Relevant to sight-reading phrases with repeated or varied figures. |
| Museformer, 2022 ([paper](https://arxiv.org/abs/2210.10349)) | Fine-grained attention to structure-related bars and coarse summaries elsewhere. | Supports bar-aware memory or retrieval over musically related bars. |
| FIGARO, 2022 ([paper](https://arxiv.org/abs/2201.10936)) | Fine-grained controllable symbolic generation with high-level descriptions and domain knowledge. | Supports explicit control vectors and interpretable musical descriptors. |
| Anticipatory Music Transformer, 2023 ([paper](https://arxiv.org/abs/2306.08620)) | Interleaves events and future/asynchronous controls for controllable symbolic generation. | Relevant to planned register, chord, density, or phrase controls that should affect later notes. |
| MuseCoco, 2023 ([paper](https://arxiv.org/abs/2306.00110)) | Text-to-attribute and attribute-to-music two-stage symbolic generation. | Supports separating user intent from note rendering. |
| GETMusic, 2023 ([paper](https://arxiv.org/abs/2305.10841)) | Unified representation and diffusion framework for generating target tracks conditioned on other tracks. | Less directly applicable to sight-reading, but reinforces track/role-aware symbolic generation. |
| BandControlNet, 2024 ([paper](https://arxiv.org/abs/2407.10462)) | Fine-grained spatiotemporal controls for popular multi-track symbolic generation. | Useful precedent for controllable density/register/texture grids. |
| SING, 2024 ([paper](https://arxiv.org/abs/2406.15647)) | Uses self-similarity guidance to improve long-term repeated structure. | Relevant to phrase repetition and variation metrics. |
| Chord-Transformer, 2025 ([paper](https://link.springer.com/article/10.1007/s40747-025-02210-2)) | Chord-controlled symbolic generation using REMI-style representation. | Reinforces explicit chord tracks as useful conditioning, though Musak should keep harmony soft. |
| MLAT, 2025 ([paper](https://link.springer.com/article/10.1186/s13636-025-00407-4)) | Multi-level attention over compound-word encoding for symbolic generation. | Recent support for compound fields plus multi-level structure. |
| CAST, 2026 ([paper](https://www.nature.com/articles/s41598-026-46750-0)) | Cascaded skeleton-to-texture framework for long-range symbolic structure. | Good watchlist item for explicit skeleton guidance; not needed before Musak's figure-family planner. |
| Musical Attention Transformer, 2026 ([paper](https://arxiv.org/abs/2605.21081)) | Music-specific attention using structural properties and metadata. | Watchlist item supporting the same broad thesis: generic attention benefits from music-aware bias. |

## Actionable Lessons For Musak

1. **Do not promote exact n-grams to tokens.** Exact `FigureNGram` identity has too much cardinality and sparsity.
2. **Factorize note prediction.** Predict token kind first, then note attributes when the kind is `note`.
3. **Add bar/cell musical coordinates.** Token position alone is not enough; the model should know bar index, position
   in bar, hand, and planned context.
4. **Introduce a planner layer.** Predict lower-entropy objects such as phrase role, chord function, register target,
   density, activity gate, figure family, and base duration.
5. **Use reference tables as priors.** Let figure statistics guide local idiom, transitions, calibration, and
   evaluation without becoming the generative vocabulary.
6. **Evaluate generated music structurally.** Add metrics for repeated figure families, phrase-level variation,
   harmonic fit, register autocorrelation, density profile, and reference alignment.

## Open Research Questions

- What is the right figure-family abstraction for sight-reading: contour/duration/property tuple, clustered embedding,
  or supervised pedagogical label?
- Should the first planner be neural, stochastic, or hybrid?
- Should phrase roles be hand-labeled, rule-derived, or inferred from repeated figure structure?
- How much hard theory should enter harmony: chord tones only, functional transitions, or empirically decoded chord
  contexts?
- How should difficulty labels interact with musical controls: as target metadata, as derived structural features, or
  as a separate curriculum objective?
