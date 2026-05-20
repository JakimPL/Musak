import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide", app_title="Tokenizer Explorer")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo

    from musak_model.decoder.notation import segment_to_score_data
    from musak_model.paths import DEFAULT_DATA_DIR, DEFAULT_PROCESSED_ROOT
    from musak_model.processing.io import load_parsed_score_json
    from musak_model.tokens.vocabulary import build_default_token_vocabulary
    from musak_shared.elements import MUSICXML_EXTENSIONS
    from musak_shared.notation.html import score_data_html
    from notebooks.utils import (
        PitchSpelling,
        default_duration_vocabulary,
        encoded_sample_to_segment,
        encoded_segments_result,
        hand_controls,
        load_encoded_shard,
        parsed_score_manifest_diagnostics,
        parsed_score_piano_roll_view_data,
        piano_roll_player_panel,
        process_score_safely,
        score_summary,
        segment_parsed_score_safely,
        segment_piano_roll_view_data,
        selected_file,
        selected_musicxml_file,
        token_rows,
    )

    return (
        DEFAULT_DATA_DIR,
        DEFAULT_PROCESSED_ROOT,
        MUSICXML_EXTENSIONS,
        Path,
        PitchSpelling,
        alt,
        build_default_token_vocabulary,
        default_duration_vocabulary,
        encoded_sample_to_segment,
        encoded_segments_result,
        hand_controls,
        load_encoded_shard,
        load_parsed_score_json,
        mo,
        parsed_score_manifest_diagnostics,
        parsed_score_piano_roll_view_data,
        piano_roll_player_panel,
        process_score_safely,
        score_summary,
        score_data_html,
        segment_parsed_score_safely,
        segment_piano_roll_view_data,
        segment_to_score_data,
        selected_file,
        selected_musicxml_file,
        token_rows,
    )


@app.cell
def _(build_default_token_vocabulary, mo):
    vocabulary_size = build_default_token_vocabulary().vocabulary_size
    title_output = mo.md(f"""
    # Tokenizer Explorer

    **Vocabulary size:** {vocabulary_size}
    """)
    title_output
    return


@app.cell
def _(
    DEFAULT_DATA_DIR,
    DEFAULT_PROCESSED_ROOT,
    Path,
    default_duration_vocabulary,
):
    def _existing_directory(path: Path) -> Path:
        current = path
        while not current.exists() or not current.is_dir():
            if current == current.parent:
                return DEFAULT_DATA_DIR

            current = current.parent

        return current

    data_root = DEFAULT_DATA_DIR
    processed_browser_path = _existing_directory(DEFAULT_PROCESSED_ROOT)
    duration_vocabulary = default_duration_vocabulary()
    return duration_vocabulary, processed_browser_path


@app.cell
def _(DEFAULT_DATA_DIR, MUSICXML_EXTENSIONS, mo, processed_browser_path):
    source_mode = mo.ui.radio(
        options={
            "Raw MusicXML": "raw",
            "Parsed JSON": "parsed",
            "Encoded JSONL": "encoded",
        },
        value="Raw MusicXML",
        inline=True,
        label="Source",
    )
    file_browser = mo.ui.file_browser(
        initial_path=DEFAULT_DATA_DIR,
        filetypes=list(MUSICXML_EXTENSIONS),
        selection_mode="file",
        multiple=False,
        label="MusicXML file",
    )
    parsed_browser = mo.ui.file_browser(
        initial_path=processed_browser_path,
        filetypes=[".json"],
        selection_mode="file",
        multiple=False,
        label="Parsed score JSON",
    )
    encoded_browser = mo.ui.file_browser(
        initial_path=processed_browser_path,
        filetypes=[".jsonl"],
        selection_mode="file",
        multiple=False,
        label="Encoded shard JSONL",
    )
    window_slider = mo.ui.slider(start=1, stop=32, step=1, value=8, label="Window bars")
    stride_slider = mo.ui.slider(start=1, stop=16, step=1, value=4, label="Stride bars")
    bpm_slider = mo.ui.slider(start=30, stop=240, step=1, value=60, label="BPM")
    prefer_flats_checkbox = mo.ui.checkbox(value=False, label="Prefer flats")
    return (
        bpm_slider,
        encoded_browser,
        file_browser,
        parsed_browser,
        prefer_flats_checkbox,
        source_mode,
        stride_slider,
        window_slider,
    )


@app.cell
def _(
    bpm_slider,
    encoded_browser,
    file_browser,
    mo,
    parsed_browser,
    prefer_flats_checkbox,
    source_mode,
    stride_slider,
    window_slider,
):
    active_browser = {
        "raw": file_browser,
        "parsed": parsed_browser,
        "encoded": encoded_browser,
    }[source_mode.value]
    browser_output = mo.vstack(
        [
            source_mode,
            active_browser,
            mo.hstack([window_slider, stride_slider, bpm_slider, prefer_flats_checkbox], gap=2),
        ],
        gap=2,
    )
    browser_output
    return


@app.cell
def _(
    encoded_browser,
    file_browser,
    parsed_browser,
    selected_file,
    selected_musicxml_file,
    source_mode,
):
    if source_mode.value == "raw":
        active_selection = selected_musicxml_file(file_browser)

    elif source_mode.value == "parsed":
        active_selection = selected_file(
            parsed_browser,
            supported_suffixes=frozenset({".json"}),
            description="parsed score JSON",
        )

    else:
        active_selection = selected_file(
            encoded_browser,
            supported_suffixes=frozenset({".jsonl"}),
            description="encoded JSONL",
        )

    selected_path = active_selection.path
    return active_selection, selected_path


@app.cell
def _(active_selection, mo):
    selection_output = None
    if active_selection.path is None:
        selection_output = mo.vstack(
            [
                mo.callout(active_selection.message or "No file selected.", kind="warn"),
                mo.md(f"Current browser value: `{active_selection.value_repr}`"),
            ],
            gap=1,
        )

    selection_output
    return


@app.cell
def _(
    encoded_sample_to_segment,
    encoded_segments_result,
    duration_vocabulary,
    load_encoded_shard,
    load_parsed_score_json,
    mo,
    parsed_score_manifest_diagnostics,
    process_score_safely,
    segment_parsed_score_safely,
    selected_path,
    source_mode,
    stride_slider,
    window_slider,
):
    if selected_path is None:
        encoded_shard = None
        processing_result = None
        processing_output = mo.md("")

    elif source_mode.value == "raw":
        encoded_shard = None
        with mo.status.spinner(title="Parsing and tokenizing selected file..."):
            processing_result = process_score_safely(
                selected_path,
                duration_vocabulary,
                window_bars=window_slider.value,
                stride_bars=stride_slider.value,
            )

        if processing_result.succeeded:
            segment_count = len(processing_result.segments)
            eligible_count = sum(segment.metadata.eligible_for_training for segment in processing_result.segments)
            callout_kind = "warn" if segment_count > 0 and eligible_count == 0 else "success"
            processing_output = mo.callout(
                (
                    f"{processing_result.path.name}: {segment_count} segment(s) produced, "
                    f"{eligible_count} eligible for training"
                ),
                kind=callout_kind,
            )
        else:
            processing_output = mo.callout(
                f"{processing_result.path.name}: {processing_result.error_type}: {processing_result.error_message}",
                kind="danger",
            )

    elif source_mode.value == "parsed":
        encoded_shard = None
        with mo.status.spinner(title="Loading parsed score..."):
            loaded_parsed_score = load_parsed_score_json(selected_path)
            parse_diagnostics = parsed_score_manifest_diagnostics(selected_path)
            processing_result = segment_parsed_score_safely(
                loaded_parsed_score,
                selected_path,
                duration_vocabulary,
                window_bars=window_slider.value,
                stride_bars=stride_slider.value,
                parse_diagnostics=parse_diagnostics,
            )
        if processing_result.succeeded:
            segment_count = len(processing_result.segments)
            eligible_count = sum(segment.metadata.eligible_for_training for segment in processing_result.segments)
            callout_kind = "warn" if segment_count > 0 and eligible_count == 0 else "success"
            processing_output = mo.callout(
                (
                    f"{selected_path.name}: parsed score loaded, {segment_count} segment(s) produced, "
                    f"{eligible_count} eligible for training"
                ),
                kind=callout_kind,
            )
        else:
            processing_output = mo.callout(
                f"{processing_result.path.name}: {processing_result.error_type}: {processing_result.error_message}",
                kind="danger",
            )
    else:
        with mo.status.spinner(title="Loading encoded shard..."):
            encoded_shard = load_encoded_shard(selected_path)
        segments = [encoded_sample_to_segment(sample, shard=encoded_shard) for sample in encoded_shard.samples]
        processing_result = encoded_segments_result(selected_path, segments=segments)
        processing_output = mo.callout(
            f"{selected_path.name}: {len(encoded_shard.samples)} encoded sample(s) loaded",
            kind="success",
        )

    processing_output
    return (processing_result,)


@app.cell
def _(mo, processing_result):
    if processing_result is None or processing_result.parse_diagnostics == "":
        diagnostics_output = mo.md("")

    else:
        diagnostics_output = mo.accordion(
            {"Parse diagnostics": mo.md(f"```text\n{processing_result.parse_diagnostics}\n```")}
        )

    diagnostics_output
    return


@app.cell
def _(mo, processing_result):
    if processing_result is None or processing_result.succeeded:
        traceback_output = mo.md("")

    else:
        traceback_output = mo.accordion({"Traceback": mo.md(f"```text\n{processing_result.traceback_text}\n```")})

    traceback_output
    return


@app.cell
def _(mo, processing_result):
    if processing_result is not None and processing_result.segments:
        segment_slider = mo.ui.slider(
            start=0,
            stop=len(processing_result.segments) - 1,
            step=1,
            value=0,
            label="Segment",
        )
        segment_slider_output = segment_slider

    else:
        segment_slider = None
        segment_slider_output = mo.md("")

    segment_slider_output
    return (segment_slider,)


@app.cell
def _(processing_result, segment_slider):
    if processing_result is not None and segment_slider is not None:
        segment = processing_result.segments[segment_slider.value]

    else:
        segment = None
    return (segment,)


@app.cell
def _(mo, segment):
    if segment is None:
        metadata_output = mo.md("")
    else:
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        features = segment.difficulty_features
        rows = [
            {"Property": "Source", "Value": str(segment.source_file)},
            {"Property": "Window start", "Value": str(segment.metadata.window_start_bar)},
            {"Property": "Key", "Value": f"{key_names[segment.scale_root]} {segment.scale_type.value}"},
            {"Property": "Time signature", "Value": f"{segment.time_numerator}/{segment.time_denominator}"},
            {"Property": "Bars", "Value": str(segment.bar_count)},
            {"Property": "Tokens", "Value": str(len(segment.tokens))},
            {
                "Property": "Difficulty level",
                "Value": str(segment.difficulty_level) if segment.difficulty_level is not None else "unlabeled",
            },
            {"Property": "Eligible for training", "Value": str(segment.metadata.eligible_for_training)},
            {
                "Property": "Ineligibility reasons",
                "Value": ", ".join(sorted(reason.value for reason in segment.metadata.ineligibility_reasons)) or "-",
            },
        ]
        if features is not None:
            rows.extend(
                [
                    {"Property": "Max RH span", "Value": str(features.max_right_hand_span_semitones)},
                    {"Property": "Max LH span", "Value": str(features.max_left_hand_span_semitones)},
                    {"Property": "Notes per beat", "Value": f"{features.notes_per_beat:.2f}"},
                    {"Property": "Voice independence", "Value": f"{features.voice_independence:.2f}"},
                ]
            )
        metadata_output = mo.ui.table(rows, selection=None, label="Segment metadata")

    metadata_output
    return


@app.cell
def _(duration_vocabulary, mo, segment, token_rows):
    if segment is None:
        tokens_output = mo.md("")
    else:
        tokens_output = mo.ui.table(
            token_rows(segment.tokens, duration_vocabulary=duration_vocabulary),
            selection=None,
            label="Unified token stream",
        )

    tokens_output
    return


@app.cell
def _(bpm_slider, duration_vocabulary, mo, score_data_html, segment, segment_to_score_data):
    if segment is None:
        notation_output = mo.md("")
    else:
        try:
            score_data = segment_to_score_data(
                segment,
                duration_vocabulary=duration_vocabulary,
                tempo=bpm_slider.value,
                measures_per_row=4,
            )
            iframe_height = f"{max(220, len(score_data.rows) * 140 + 24)}px"
            notation_output = mo.iframe(score_data_html(score_data), height=iframe_height)
        except ValueError as exception:
            notation_output = mo.callout(f"Notation rendering unavailable: {exception}", kind="warn")

    notation_output
    return


@app.cell
def _(mo, processing_result, segment):
    if processing_result is not None and processing_result.parsed_score is not None and segment is not None:
        piano_roll_scope = mo.ui.radio(
            options={
                "Selected segment": "segment",
                "Full score": "full_score",
            },
            value="Selected segment",
            inline=True,
            label="Piano roll source",
        )
        piano_roll_scope_output = piano_roll_scope

    else:
        piano_roll_scope = None
        piano_roll_scope_output = mo.md("")

    piano_roll_scope_output
    return (piano_roll_scope,)


@app.cell
def _(hand_controls, mo):
    piano_roll_hand_controls = hand_controls(mo)
    return (piano_roll_hand_controls,)


@app.cell
def _(
    PitchSpelling,
    alt,
    bpm_slider,
    duration_vocabulary,
    mo,
    parsed_score_piano_roll_view_data,
    piano_roll_hand_controls,
    piano_roll_player_panel,
    piano_roll_scope,
    prefer_flats_checkbox,
    processing_result,
    segment,
    segment_piano_roll_view_data,
):
    pitch_spelling = PitchSpelling.FLATS if prefer_flats_checkbox.value else PitchSpelling.SHARPS
    if (
        processing_result is not None
        and processing_result.parsed_score is not None
        and (piano_roll_scope is None or piano_roll_scope.value == "full_score")
    ):
        view_data = parsed_score_piano_roll_view_data(
            processing_result.parsed_score,
            pitch_spelling=pitch_spelling,
            bpm=bpm_slider.value,
        )

    elif segment is not None:
        view_data = segment_piano_roll_view_data(
            segment,
            duration_vocabulary=duration_vocabulary,
            pitch_spelling=pitch_spelling,
            bpm=bpm_slider.value,
        )

    else:
        view_data = None

    piano_roll_output = piano_roll_player_panel(
        view_data,
        mo=mo,
        alt=alt,
        bpm=bpm_slider.value,
        controls=piano_roll_hand_controls,
    )

    piano_roll_output
    return


@app.cell
def _(duration_vocabulary, mo, score_summary, segment):
    if segment is None:
        score_output = mo.md("")
    else:
        score_output = mo.ui.table(
            score_summary(segment, duration_vocabulary=duration_vocabulary),
            selection=None,
            label="Decoded music21 score summary",
        )

    score_output
    return


if __name__ == "__main__":
    app.run()
