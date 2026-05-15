import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide", app_title="Tokenizer Explorer")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo

    from musak_model.common.elements import MIDI_MAX_PITCH
    from musak_model.paths import DEFAULT_DATA_DIR, DEFAULT_PROCESSED_ROOT
    from musak_model.processing.io import load_parsed_score_json
    from musak_model.tokens.vocabulary import build_default_token_vocabulary
    from notebooks.utils import (
        PitchSpelling,
        default_duration_vocabulary,
        encoded_sample_to_segment,
        encoded_segments_result,
        load_encoded_shard,
        parsed_score_manifest_diagnostics,
        parsed_score_piano_roll_dataframe,
        piano_roll_dataframe,
        process_score_safely,
        score_summary,
        segment_parsed_score_safely,
        selected_file,
        selected_musicxml_file,
        token_rows,
    )

    return (
        MIDI_MAX_PITCH,
        DEFAULT_DATA_DIR,
        DEFAULT_PROCESSED_ROOT,
        Path,
        PitchSpelling,
        alt,
        build_default_token_vocabulary,
        default_duration_vocabulary,
        encoded_sample_to_segment,
        encoded_segments_result,
        load_encoded_shard,
        load_parsed_score_json,
        mo,
        parsed_score_piano_roll_dataframe,
        parsed_score_manifest_diagnostics,
        piano_roll_dataframe,
        process_score_safely,
        score_summary,
        selected_file,
        selected_musicxml_file,
        segment_parsed_score_safely,
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
def _(DEFAULT_DATA_DIR, DEFAULT_PROCESSED_ROOT, Path, default_duration_vocabulary):
    def _existing_directory(path: Path) -> Path:
        current = path
        while not current.exists() or not current.is_dir():
            if current == current.parent:
                return DEFAULT_DATA_DIR
            current = current.parent
        return current

    data_root = DEFAULT_DATA_DIR
    pdmx_root = data_root / "PDMX" / "mxl"
    initial_browser_path = pdmx_root if pdmx_root.exists() else data_root
    processed_browser_path = _existing_directory(DEFAULT_PROCESSED_ROOT)
    duration_vocabulary = default_duration_vocabulary()
    return (
        duration_vocabulary,
        initial_browser_path,
        processed_browser_path,
    )


@app.cell
def _(initial_browser_path, mo, processed_browser_path):
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
        initial_path=initial_browser_path,
        filetypes=[".mxl", ".musicxml", ".xml"],
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
def _(encoded_browser, file_browser, parsed_browser, selected_file, selected_musicxml_file, source_mode):
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
    duration_vocabulary,
    encoded_sample_to_segment,
    encoded_segments_result,
    load_encoded_shard,
    load_parsed_score_json,
    mo,
    parsed_score_manifest_diagnostics,
    process_score_safely,
    selected_path,
    segment_parsed_score_safely,
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
    return encoded_shard, processing_result


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
            {"Property": "Key", "Value": f"{key_names[segment.key_root]} {segment.scale_type.value}"},
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
def _(
    MIDI_MAX_PITCH,
    PitchSpelling,
    alt,
    bpm_slider,
    duration_vocabulary,
    mo,
    parsed_score_piano_roll_dataframe,
    piano_roll_dataframe,
    prefer_flats_checkbox,
    processing_result,
    segment,
):
    pitch_spelling = PitchSpelling.FLATS if prefer_flats_checkbox.value else PitchSpelling.SHARPS
    if segment is not None:
        piano_roll_df = piano_roll_dataframe(
            segment,
            duration_vocabulary=duration_vocabulary,
            pitch_spelling=pitch_spelling,
            bpm=bpm_slider.value,
        )
        piano_roll_title = "Decoded segment piano roll"
        measure_duration = segment.time_numerator / segment.time_denominator
        bar_domain = [segment.metadata.window_start_bar + 1, segment.metadata.window_start_bar + segment.bar_count + 1]
        seconds_domain = [0.0, segment.bar_count * measure_duration * 4 * 60 / bpm_slider.value]
    elif processing_result is not None and processing_result.parsed_score is not None:
        piano_roll_parsed_score = processing_result.parsed_score
        piano_roll_df = parsed_score_piano_roll_dataframe(
            piano_roll_parsed_score,
            pitch_spelling=pitch_spelling,
            bpm=bpm_slider.value,
        )
        piano_roll_title = "Parsed score piano roll"
        parsed_bar_count = max(
            len(piano_roll_parsed_score.right_hand_bars),
            len(piano_roll_parsed_score.left_hand_bars),
        )
        measure_duration = piano_roll_parsed_score.time_numerator / piano_roll_parsed_score.time_denominator
        bar_domain = [1, parsed_bar_count + 1]
        seconds_domain = [0.0, parsed_bar_count * measure_duration * 4 * 60 / bpm_slider.value]
    else:
        piano_roll_df = None
        piano_roll_title = ""
        bar_domain = [0, 1]
        seconds_domain = [0.0, 1.0]

    if piano_roll_df is None:
        piano_roll_output = mo.md("")
    elif piano_roll_df.empty:
        piano_roll_output = mo.callout("No note events decoded for this score.", kind="warn")
    else:
        sharp_pitch_names = ["C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-"]
        flat_pitch_names = ["C-", "Db", "D-", "Eb", "E-", "F-", "Gb", "G-", "Ab", "A-", "Bb", "B-"]
        pitch_names = flat_pitch_names if prefer_flats_checkbox.value else sharp_pitch_names
        pitch_label_expression = f"{pitch_names}[datum.value % 12] + floor(datum.value / 12 - 1)"
        y_domain = [
            max(0, float(piano_roll_df["midi_pitch"].min()) - 1),
            min(MIDI_MAX_PITCH, float(piano_roll_df["midi_pitch"].max()) + 1),
        ]
        note_bars = (
            alt.Chart(piano_roll_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "bar_start:Q",
                    title="Bars",
                    axis=alt.Axis(grid=True),
                    scale=alt.Scale(domain=bar_domain),
                ),
                x2="bar_end:Q",
                y=alt.Y(
                    "midi_pitch:Q",
                    title="Pitch",
                    axis=alt.Axis(labelExpr=pitch_label_expression),
                    scale=alt.Scale(domain=y_domain),
                ),
                color=alt.Color("hand:N", title="Hand"),
                tooltip=[
                    alt.Tooltip("hand:N", title="Hand"),
                    alt.Tooltip("pitch:N", title="Pitch"),
                    alt.Tooltip("midi_pitch:Q", title="MIDI"),
                    alt.Tooltip("bar_start:Q", title="Bar start", format=".3f"),
                    alt.Tooltip("bar_end:Q", title="Bar end", format=".3f"),
                    alt.Tooltip("start_seconds:Q", title="Start (s)", format=".3f"),
                    alt.Tooltip("duration_fraction:N", title="Duration"),
                    alt.Tooltip("duration_seconds:Q", title="Duration (s)", format=".3f"),
                    alt.Tooltip("token:N", title="Token"),
                    alt.Tooltip("token_index:Q", title="Token index"),
                ],
            )
        )
        seconds_axis = (
            alt.Chart(piano_roll_df)
            .mark_rule(opacity=0)
            .encode(
                x=alt.X(
                    "start_seconds:Q",
                    title="Time (s)",
                    axis=alt.Axis(orient="top", grid=False),
                    scale=alt.Scale(domain=seconds_domain),
                )
            )
        )
        chart = (
            alt.layer(note_bars, seconds_axis)
            .resolve_scale(x="independent")
            .properties(width="container", height=400, title=piano_roll_title)
        )
        piano_roll_output = mo.ui.altair_chart(chart)
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
