import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide", app_title="Tokenizer Explorer")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo

    from musak_model.tokens.vocabulary import VOCAB_SIZE
    from notebooks.utils import (
        default_duration_vocabulary,
        parsed_score_piano_roll_dataframe,
        piano_roll_dataframe,
        process_score_safely,
        score_summary,
        selected_musicxml_file,
        token_rows,
    )

    return (
        Path,
        VOCAB_SIZE,
        alt,
        default_duration_vocabulary,
        mo,
        parsed_score_piano_roll_dataframe,
        piano_roll_dataframe,
        process_score_safely,
        score_summary,
        selected_musicxml_file,
        token_rows,
    )


@app.cell
def _(VOCAB_SIZE, mo):
    title_output = mo.md(f"""
    # Tokenizer Explorer

    **Vocabulary size:** {VOCAB_SIZE}
    """)
    title_output
    return


@app.cell
def _(Path, default_duration_vocabulary):
    data_root = Path("data")
    pdmx_root = data_root / "PDMX" / "mxl"
    initial_browser_path = pdmx_root if pdmx_root.exists() else data_root
    duration_vocabulary = default_duration_vocabulary()
    return duration_vocabulary, initial_browser_path


@app.cell
def _(initial_browser_path, mo):
    file_browser = mo.ui.file_browser(
        initial_path=initial_browser_path,
        filetypes=[".mxl", ".musicxml", ".xml"],
        selection_mode="file",
        multiple=False,
        label="MusicXML file",
    )
    window_slider = mo.ui.slider(start=1, stop=32, step=1, value=8, label="Window bars")
    stride_slider = mo.ui.slider(start=1, stop=16, step=1, value=4, label="Stride bars")
    browser_output = mo.vstack([file_browser, mo.hstack([window_slider, stride_slider], gap=2)], gap=2)
    browser_output
    return file_browser, stride_slider, window_slider


@app.cell
def _(file_browser, selected_musicxml_file):
    selected_file = selected_musicxml_file(file_browser)
    selected_path = selected_file.path
    return selected_file, selected_path


@app.cell
def _(mo, selected_file):
    if selected_file.path is None:
        selection_output = mo.vstack(
            [
                mo.callout(selected_file.message or "No file selected.", kind="warn"),
                mo.md(f"Current browser value: `{selected_file.value_repr}`"),
            ],
            gap=1,
        )
    else:
        selection_output = mo.callout(f"Selected file: `{selected_file.path}`", kind="neutral")
    selection_output
    return


@app.cell
def _(mo, process_score_safely, selected_path, stride_slider, window_slider):
    if selected_path is None:
        processing_result = None
        processing_output = mo.md("")
    else:
        with mo.status.spinner(title="Parsing and tokenizing selected file..."):
            processing_result = process_score_safely(
                selected_path,
                window_bars=window_slider.value,
                stride_bars=stride_slider.value,
            )

        if processing_result.succeeded:
            processing_output = mo.callout(
                f"{processing_result.path.name}: {len(processing_result.segments)} segment(s) produced",
                kind="success",
            )
        else:
            processing_output = mo.callout(
                f"{processing_result.path.name}: {processing_result.error_type}: {processing_result.error_message}",
                kind="danger",
            )
    processing_output
    return (processing_result,)


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
            {"Property": "Unified tokens", "Value": str(len(segment.tokens))},
            {"Property": "RH tokens", "Value": str(len(segment.right_hand_tokens))},
            {"Property": "LH tokens", "Value": str(len(segment.left_hand_tokens))},
            {"Property": "Difficulty level", "Value": str(segment.difficulty_level or "unlabeled")},
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
    alt, duration_vocabulary, mo, parsed_score_piano_roll_dataframe, piano_roll_dataframe, processing_result, segment
):
    if segment is not None:
        piano_roll_df = piano_roll_dataframe(segment, duration_vocabulary=duration_vocabulary)
        piano_roll_title = "Decoded segment piano roll"
    elif processing_result is not None and processing_result.parsed_score is not None:
        piano_roll_df = parsed_score_piano_roll_dataframe(processing_result.parsed_score)
        piano_roll_title = "Parsed score piano roll"
    else:
        piano_roll_df = None
        piano_roll_title = ""

    if piano_roll_df is None:
        piano_roll_output = mo.md("")
    elif piano_roll_df.empty:
        piano_roll_output = mo.callout("No note events decoded for this score.", kind="warn")
    else:
        chart = (
            alt.Chart(piano_roll_df)
            .mark_bar()
            .encode(
                x=alt.X("start:Q", title="Start"),
                x2="end:Q",
                y=alt.Y("midi_pitch:Q", title="MIDI pitch"),
                color=alt.Color("hand:N", title="Hand"),
                tooltip=["hand", "midi_pitch", "start", "duration"],
            )
            .properties(width="container", height=360, title=piano_roll_title)
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
