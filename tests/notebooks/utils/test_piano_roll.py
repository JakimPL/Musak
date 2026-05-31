from fractions import Fraction
from pathlib import Path

import altair as alt

from musak_model.data.schema import ParsedBar, ParsedNote, ParsedScore, Segment, SegmentMetadata
from musak_model.data.tokenization_context import tokenization_context_from_scale
from musak_model.decoder import encoded_exercise_to_segment
from musak_model.processing.io import append_jsonl, write_json_model
from musak_model.processing.snapshot import build_tokenizer_snapshot
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import Hand, HandToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise
from notebooks.utils.encoded import load_encoded_shard
from notebooks.utils.piano_roll import (
    ChordHighlight,
    PitchSpelling,
    filter_piano_roll_dataframe,
    midi_pitch_name,
    parsed_score_piano_roll_dataframe,
    parsed_score_piano_roll_view_data,
    piano_roll_chart,
    piano_roll_dataframe,
    scale_pitch_class_set,
    segment_piano_roll_view_data,
)


def _two_hand_score() -> ParsedScore:
    return ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0))],
            )
        ],
        left_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0))],
            )
        ],
    )


def test_midi_pitch_name_uses_scientific_pitch_octaves() -> None:
    assert midi_pitch_name(60) == "C-4"
    assert midi_pitch_name(61) == "C#4"
    assert midi_pitch_name(61, pitch_spelling=PitchSpelling.FLATS) == "Db4"
    assert midi_pitch_name(58, pitch_spelling=PitchSpelling.FLATS) == "Bb3"


def test_segment_piano_roll_dataframe_includes_axis_and_token_fields(duration_vocabulary: DurationVocabulary) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=1, octave_offset=0, duration_id=quarter_id),
        ],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=2,
            source_file=Path("score.mxl"),
        ),
    )

    row = piano_roll_dataframe(
        segment,
        duration_vocabulary=duration_vocabulary,
        pitch_spelling=PitchSpelling.SHARPS,
        bpm=120,
    ).iloc[0]

    assert row["pitch"] == "C#5"
    assert row["bar_start"] == 3.0
    assert row["bar_end"] == 3.25
    assert row["bar_start_display"] == 3.004
    assert row["bar_end_display"] == 3.246
    assert row["start_seconds"] == 0.0
    assert row["duration_fraction"] == "1:4"
    assert row["duration_seconds"] == 0.5
    assert row["token_index"] == 1
    assert row["token"] == "1♯(1:4)"


def test_segment_piano_roll_view_data_includes_events_domains_and_frame(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
        ],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=2,
            window_start_bar=3,
            source_file=Path("score.mxl"),
        ),
    )

    view_data = segment_piano_roll_view_data(segment, duration_vocabulary=duration_vocabulary, bpm=120)

    assert len(view_data.events) == 1
    assert view_data.bar_domain == (4.0, 6.0)
    assert view_data.seconds_domain == (0.0, 4.0)
    assert view_data.dataframe.iloc[0]["pitch"] == "C-5"


def test_parsed_score_piano_roll_dataframe_uses_pitch_spelling_without_token_fields() -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=61, duration=Fraction(1, 4), beat_offset=Fraction(1, 4))],
            )
        ],
        left_hand_bars=[],
    )

    row = parsed_score_piano_roll_dataframe(score, pitch_spelling=PitchSpelling.FLATS, bpm=60).iloc[0]

    assert row["pitch"] == "Db4"
    assert row["bar_start"] == 1.25
    assert row["bar_end"] == 1.5
    assert row["start_seconds"] == 1.0
    assert row["duration_fraction"] == "1:4"
    assert row["duration_seconds"] == 1.0
    assert row["token"] is None


def test_parsed_score_piano_roll_view_data_and_filtering() -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=61, duration=Fraction(1, 4), beat_offset=Fraction(0))],
            )
        ],
        left_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0))],
            )
        ],
    )

    view_data = parsed_score_piano_roll_view_data(score, pitch_spelling=PitchSpelling.FLATS, bpm=60)
    left_frame = filter_piano_roll_dataframe(view_data.dataframe, hands=frozenset({Hand.LEFT}))

    assert view_data.bar_domain == (1.0, 2.0)
    assert view_data.seconds_domain == (0.0, 4.0)
    assert left_frame["hand"].tolist() == ["left"]
    assert left_frame["midi_pitch"].tolist() == [48]


def test_piano_roll_chart_uses_fixed_hand_colors_and_note_outlines() -> None:
    score = ParsedScore(
        scale_root=0,
        key_fifths=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
        right_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=60, duration=Fraction(1, 4), beat_offset=Fraction(0))],
            )
        ],
        left_hand_bars=[
            ParsedBar(
                time_numerator=4,
                time_denominator=4,
                key_fifths=0,
                events=[ParsedNote(midi_pitch=48, duration=Fraction(1, 4), beat_offset=Fraction(0))],
            )
        ],
    )

    chart = piano_roll_chart(parsed_score_piano_roll_view_data(score), alt=alt)
    note_layer = chart.to_dict()["layer"][0]

    assert note_layer["mark"]["stroke"] == "#ffffff"
    assert note_layer["mark"]["strokeWidth"] == 0.7
    assert note_layer["encoding"]["x"]["field"] == "bar_start_display"
    assert note_layer["encoding"]["x2"]["field"] == "bar_end_display"
    assert note_layer["encoding"]["color"]["scale"] == {
        "domain": ["left", "right"],
        "range": ["#1f77b4", "#ff7f0e"],
    }


def test_scale_pitch_class_set_is_root_transposed_and_modal() -> None:
    assert scale_pitch_class_set(0, ScaleType.MAJOR) == frozenset({0, 2, 4, 5, 7, 9, 11})
    assert scale_pitch_class_set(2, ScaleType.MAJOR) == frozenset({2, 4, 6, 7, 9, 11, 1})
    # A harmonic minor raises the seventh degree (G#).
    assert scale_pitch_class_set(9, ScaleType.HARMONIC_MINOR) == frozenset({9, 11, 0, 2, 4, 5, 8})


def test_piano_roll_chart_layers_scale_and_chord_highlights_behind_notes() -> None:
    view_data = parsed_score_piano_roll_view_data(_two_hand_score())
    chord_highlights = (
        ChordHighlight(start_in_bars=1.0, end_in_bars=2.0, pitch_classes=frozenset({0, 4, 7}), label="I"),
    )

    layers = piano_roll_chart(
        view_data,
        alt=alt,
        scale_pitch_classes=scale_pitch_class_set(0, ScaleType.MAJOR),
        chord_highlights=chord_highlights,
    ).to_dict()["layer"]

    assert len(layers) == 4
    scale_layer, chord_layer, note_layer = layers[0], layers[1], layers[2]
    # Scale band: full-width (no x), one semitone tall, fixed green fill with a separating outline.
    assert scale_layer["mark"]["type"] == "rect"
    assert scale_layer["mark"]["fill"] == "#43a047"
    assert scale_layer["mark"]["stroke"] == "#2e7d32"
    assert "x" not in scale_layer["encoding"]
    assert scale_layer["encoding"]["y"]["field"] == "pitch_low"
    assert scale_layer["encoding"]["y2"]["field"] == "pitch_high"
    # Chord band: windowed in time (x/x2), distinct purple fill.
    assert chord_layer["mark"]["type"] == "rect"
    assert chord_layer["mark"]["fill"] == "#7e57c2"
    assert chord_layer["encoding"]["x"]["field"] == "start_in_bars"
    assert chord_layer["encoding"]["x2"]["field"] == "end_in_bars"
    # Notes stay drawn on top of both highlight bands.
    assert note_layer["encoding"]["x"]["field"] == "bar_start_display"


def test_piano_roll_chart_omits_highlight_layers_when_not_provided() -> None:
    chart = piano_roll_chart(parsed_score_piano_roll_view_data(_two_hand_score()), alt=alt)

    assert len(chart.to_dict()["layer"]) == 2


def test_load_encoded_shard_rebuilds_token_vocabulary(
    tmp_path: Path,
    tokenization_config: TokenizationConfig,
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    snapshot = build_tokenizer_snapshot(
        tokenization_config,
        duration_vocabulary=duration_vocabulary,
        token_vocabulary=token_vocabulary,
    )
    shard_path = tmp_path / "encoded" / snapshot.tokenizer_hash / "data-00000.jsonl"
    write_json_model(snapshot, shard_path.parent / "tokenizer.json", overwrite=True)
    sample = EncodedExercise(
        token_ids=token_vocabulary.encode([HandToken(hand=Hand.RIGHT)]),
        bar_positions=[0],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            tokenization_context=tokenization_context_from_scale(scale_root=0, scale_type=ScaleType.MAJOR),
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("piece.mxl"),
        ),
    )
    append_jsonl(sample, shard_path)

    shard = load_encoded_shard(shard_path)

    assert shard.snapshot.tokenizer_hash == snapshot.tokenizer_hash
    assert shard.samples == [sample]
    assert encoded_exercise_to_segment(shard.samples[0], token_vocabulary=shard.token_vocabulary).tokens == [
        HandToken(hand=Hand.RIGHT)
    ]
