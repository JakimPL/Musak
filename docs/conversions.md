# Conversion Index

This page lists every conversion function in the repository — operations whose primary job is to transform between
representations or coordinate systems. Consult it before adding a new conversion (per
`docs/guidelines.md` §Shared Ownership rule 9): reuse an existing primitive rather than re-inlining arithmetic, and
place new conversions according to the rule.

## Universal music primitives — `musak_shared`

Operations that are model-implementation-agnostic and depend on no sibling package.

| Function | File | Input → Output |
| --- | --- | --- |
| `pitch_class_from_key_fifths` | `musak_shared/elements.py` | key fifths → pitch class |
| `key_fifths_from_pitch_class` | `musak_shared/elements.py` | pitch class → key fifths |
| `get_note_name` | `musak_shared/names.py` | MIDI → note name |
| `midi_to_vexflow_key` | `musak_shared/names.py` | MIDI → VexFlow key (with flats/sharps preference) |
| `format_ratio` | `musak_shared/ratios.py` | `Fraction` / `(num, den)` → `"num/den"` text |
| `parse_ratio` | `musak_shared/ratios.py` | `"num/den"` text → `Fraction` |

## Token-coordinate primitives — `musak_model/tokens`

Operations on the model's token coordinates (`degree`, `accidental`, `octave_offset`, `scale_size`, `ScaleType`,
`Hand`, `NoteToken`).

| Function | File | Input → Output |
| --- | --- | --- |
| `scale_size_for_type` | `tokens/schema.py` | `ScaleType` → scale degree count |
| `degree_pitch_class` | `tokens/pitch.py` | `(degree, accidental, scale_type)` → pitch class (mod 12) |
| `note_token_to_midi_pitch` | `tokens/pitch.py` | `NoteToken` + `scale_root` + `scale_type` + `hand` → MIDI |
| `note_diatonic_position` | `tokens/pitch.py` | `NoteToken` + `scale_size` → diatonic position |
| `diatonic_position_to_degree_and_octave` | `tokens/pitch.py` | diatonic position + `scale_size` → `(degree, octave_offset)` |

### Vocabulary encoders / decoders

| Function | File | Input → Output |
| --- | --- | --- |
| `TokenVocabulary.token_to_id` / `.id_to_token` | `tokens/vocabulary.py` | `Token` ↔ token id |
| `TokenVocabulary.encode` / `.decode` | `tokens/vocabulary.py` | `list[Token]` ↔ `list[int]` |
| `DurationVocabulary.fraction_to_id` / `.id_to_fraction` | `tokens/duration.py` | `Fraction` ↔ duration id |
| `DurationVocabulary.duration_id_or_none` / `.require_duration_id` / `.find_closest` | `tokens/duration.py` | `Fraction` → duration id (variants) |
| `duration_tick_denominator` | `tokens/duration.py` | duration vocabulary → shared integer tick denominator |
| `duration_fraction_to_ticks` | `tokens/duration.py` | `Fraction` + denominator → integer ticks |
| `tokens_to_text` / `tokens_from_text` / `token_from_text` | `tokens/text.py` | `Token` sequence ↔ string |

## Composite conversions in their downstream domains

Every entry lists the two entities being bridged and the package that owns the conversion.

### Data ingestion — MIDI → tokens (`musak_model/data`)

| Function | File | Input → Output |
| --- | --- | --- |
| `pitch_to_degree` | `data/converter.py` | MIDI + pitch-set `scale_root` + spelling `key_fifths` + `scale_type` + `hand` → `PitchDegree` |
| `note_to_token` | `data/segmenter/bar.py` | `ParsedNote` → `NoteToken` |
| `chord_to_tokens` | `data/segmenter/bar.py` | `ParsedChord` → `list[Token]` |
| `tokens_from_bar_groups` | `data/segmenter/bar.py` | `list[TimedTokenGroup]` → `list[Token]` |
| `normalize_bar_events`, `tokenize_event` | `data/segmenter/bar.py` | parsed bar → tokens |
| `segment_score` | `data/segmenter/segmenter.py` | `ParsedScore` → `list[Segment]` |
| `parse_score` | `data/parser.py` | MusicXML path → `ParsedScore` |
| `quantize_duration_to_id` / `quantize_duration` | `data/quantizer.py` | `Fraction` → duration id |
| `pitch_class_histogram`, `normalized_histogram`, `ranked_candidates` | `data/scale_matcher/` | parsed bars → scale candidates |

### Figure extraction — tokens → figure (`musak_model/n_grams`)

| Function | File | Input → Output |
| --- | --- | --- |
| `extract_hand_onset_runs` | `n_grams/figure/parser.py` | `list[Token]` → `dict[Hand, tuple[HandOnsetRun, ...]]` |
| `build_figure_ngram`, `build_figure_ngrams_from_run`, `build_figure_ngrams_from_runs` | `n_grams/figure/builder.py` | `NoteTokens` / `HandOnsetRun` → `FigureNGram` |
| `iter_figure_occurrences_from_run`, `iter_figure_signatures_from_run` | `n_grams/figure/signature.py` | `HandOnsetRun` → `FigureOccurrence` / `FigureSignature` |
| `build_figure_signature_from_raw_window` | `n_grams/figure/signature.py` | raw onsets → `FigureSignature` |
| `figure_signature_to_ngram` | `n_grams/figure/signature.py` | `FigureSignature` → `FigureNGram` |
| `figure_signature_to_json` / `figure_signature_from_json` | `n_grams/figure/signature.py` | `FigureSignature` ↔ JSON string |
| `count_segment_rhythm_metrics`, `_fraction_text`, `_alignment_value`, `_rhythm_ngram_value` | `n_grams/profile/rhythm/extraction.py` | `Segment` → rhythm count entries |

### Synthetic generation — figure → tokens, chord → tones (`musak_model/synthetic`)

| Function | File | Input → Output |
| --- | --- | --- |
| `expand_chord_to_tones` | `synthetic/harmony/expansion.py` | `Chord` + `scale_type` + vocabulary → `tuple[ChordTone, ...]` |
| `chord_pitch_class_set` | `synthetic/harmony/expansion.py` | `Chord` + `scale_type` + vocabulary → `frozenset[pitch_class]` |
| `chord_window_grid` | `synthetic/harmony/windows.py` | `(measure_duration, total_duration, resolution)` → `tuple[(start, end), ...]` (bar-aligned, barline-truncated) |
| `anchor_figure_to_tokens` | `synthetic/substitution/emission.py` | `FigureNGram` + anchor + `base_duration` → `list[Token]` |

### Token factorization (`musak_model/tokens`)

| Function | File | Input → Output |
| --- | --- | --- |
| `token_to_attributes`, `token_id_to_attributes` | `tokens/factorized.py` | `Token` / flat token id → `TokenAttributes` |
| `attributes_to_token`, `predicted_attributes_to_token` | `tokens/factorized.py` | strict target attributes / model-style predicted attributes → `Token` |
| `attributes_to_token_id`, `predicted_attributes_to_token_id` | `tokens/factorized.py` | strict target attributes / model-style predicted attributes → flat token id |

### Musical auxiliary targets (`musak_model/auxiliary`)

| Function | File | Input → Output |
| --- | --- | --- |
| `musical_auxiliary_target_ids_from_difficulty_features` | `auxiliary/targets.py` | `DifficultyFeatures` / missing features + configured bucket boundaries → auxiliary target ids |
| `musical_auxiliary_target_tensors_from_ids` | `auxiliary/targets.py` | auxiliary target ids → scalar target tensors |
| `stack_musical_auxiliary_targets` | `auxiliary/targets.py` | per-example auxiliary target tensors → batched target tensors |

### Decoding to notation / piano-roll (`musak_model/decoder`)

| Function | File | Input → Output |
| --- | --- | --- |
| `segment_to_piano_roll_events`, `tokens_to_piano_roll_events`, `parsed_score_to_piano_roll_events` | `decoder/piano_roll.py` | `Segment` / `list[Token]` / `ParsedScore` → `list[PianoRollEvent]` |
| `segment_to_music21_score` | `decoder/music21.py` | `Segment` → `music21.stream.Score` |
| `segment_to_score_data`, `segment_to_notation_events` | `decoder/notation.py` | `Segment` → VexFlow `ScoreData` / notation events |
| `note_token_to_vexflow_spelling` | `decoder/notation.py` | `NoteToken` + MIDI + pitch-set `scale_root` + spelling `key_fifths` → `VexflowSpelling` |
| `segment_key_signature_name`, `segment_spelling_key_fifths` | `decoder/notation.py` | `Segment` metadata tokenization context → notation key signature / spelling key fifths |
| `key_signature_name`, `key_fifths_for_scale` | `decoder/notation.py` | fallback `(scale_root, scale_type)` → key-signature name / key fifths |
| `_letter_for_pitch_class_in_key_signature`, `_accidental_for_letter_pitch_class`, `_vexflow_octave` | `decoder/notation.py` | helper conversions for VexFlow spelling |

### Other domain-specific conversions

| Function | File | Input → Output |
| --- | --- | --- |
| `EncodedExercise.to_segment`, `encoded_exercise_to_segment` | `training/ingestion/schema.py`, `decoder/encoded.py` | `EncodedExercise` → `Segment` |
| `state_from_tokens`, `state_from_token_ids` | `generation/constraints.py` | `list[Token]` / `list[int]` → `GenerationConstraintState` |
| `decoder_input_coordinates_from_tokens`, `decoder_input_coordinates_from_token_ids` | `generation/coordinates.py` | token prefix / token-id prefix + generation constraints → `DecoderInputCoordinates` |
| `segment_from_tokens`, `constraints_from_config`, `scale_type_to_id`, `bar_positions` | `evaluation/generation/sampling.py` | evaluation-side conversions |

## When in doubt

If a conversion does not clearly fit any row above, it is probably a new composite — place it next to whichever
entity its caller already owns, and add a row here in the same change.
