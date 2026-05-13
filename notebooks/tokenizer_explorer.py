import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", app_title="Tokenizer Explorer")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Tokenizer Explorer

    Explore how MusicXML files are parsed and tokenized into scale-degree
    token sequences for model training.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    from musak_model.tokens.vocabulary import VOCAB_SIZE

    data_root = Path("data")
    available_folders = sorted(folder for folder in data_root.iterdir() if folder.is_dir())
    return VOCAB_SIZE, available_folders, data_root, Path


@app.cell
def _(VOCAB_SIZE, available_folders, mo):
    folder_picker = mo.ui.dropdown(
        options={folder.name: folder for folder in available_folders},
        value=available_folders[0].name,
        label="Data folder",
    )
    mo.md(f"**Vocabulary size:** {VOCAB_SIZE} tokens\n\n{folder_picker}")
    return (folder_picker,)


@app.cell
def _(folder_picker):
    selected_folder = folder_picker.value
    mxl_files = sorted(selected_folder.glob("*.mxl"))
    return mxl_files, selected_folder


@app.cell
def _(mo, mxl_files):
    file_picker = mo.ui.dropdown(
        options={path.name[:24] + "…": path for path in mxl_files},
        value=next(iter(mxl_files)).name[:24] + "…",
        label="Score file",
    )
    window_slider = mo.ui.slider(start=2, stop=16, step=2, value=4, label="Window (bars)")
    stride_slider = mo.ui.slider(start=1, stop=8, step=1, value=2, label="Stride (bars)")
    mo.hstack([file_picker, window_slider, stride_slider], gap=2)
    return file_picker, stride_slider, window_slider


@app.cell
def _(file_picker, mo, stride_slider, window_slider):
    from musak_model.data.config import SegmentationConfig
    from musak_model.data.pipeline import process_file

    selected_path = file_picker.value

    with mo.status.spinner(title="Parsing & tokenizing…"):
        segments = process_file(
            selected_path,
            segmentation=SegmentationConfig(
                window_bars=window_slider.value,
                stride_bars=stride_slider.value,
            ),
        )

    mo.callout(
        mo.md(f"**{selected_path.name}** — {len(segments)} segment(s) produced"),
        kind="success",
    )
    return process_file, segments, selected_path


@app.cell
def _(mo, segments):
    segment_picker = mo.ui.slider(
        start=0,
        stop=len(segments) - 1,
        step=1,
        value=0,
        label=f"Segment (0 – {len(segments) - 1})",
    )
    segment_picker
    return (segment_picker,)


@app.cell
def _(segment_picker, segments):
    segment = segments[segment_picker.value]
    return (segment,)


@app.cell
def _(mo, segment):
    from musak_model.tokens.schema import BarToken, EndToken, NoteToken, RestToken

    def _token_label(token: object) -> str:
        if isinstance(token, NoteToken):
            acc = {-1: "b", 0: "", 1: "#"}[token.accidental]
            reg = f"[{token.octave_offset:+d}]" if token.octave_offset != 0 else ""
            return f"{token.degree}{acc}{reg}/{token.duration.value[:2]}"
        if isinstance(token, RestToken):
            return f"r/{token.duration.value[:2]}"
        if isinstance(token, BarToken):
            return "|"
        if isinstance(token, EndToken):
            return "‖"
        return "?"

    def _render_hand(tokens: list, label: str) -> object:
        bars: list[list[str]] = []
        current: list[str] = []
        for token in tokens:
            if isinstance(token, BarToken):
                bars.append(current)
                current = []
            elif isinstance(token, EndToken):
                break
            else:
                current.append(_token_label(token))

        rows = []
        for bar_index, bar_tokens in enumerate(bars):
            cells = " · ".join(bar_tokens) if bar_tokens else "∅"
            rows.append(mo.hstack([mo.md(f"**{bar_index + 1}**"), mo.md(cells)], gap=1))

        return mo.vstack([mo.md(f"### {label}"), *rows], gap=0)

    rh_panel = _render_hand(segment.right_hand_tokens, "Right hand")
    lh_panel = _render_hand(segment.left_hand_tokens, "Left hand")
    mo.hstack([rh_panel, lh_panel], gap=4, justify="start")
    return BarToken, EndToken, NoteToken, RestToken


@app.cell
def _(mo, segment):
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_name = key_names[segment.key_root]

    meta_table = mo.ui.table(
        data=[
            {"Property": "Key", "Value": f"{key_name} {segment.scale_type.value}"},
            {
                "Property": "Time signature",
                "Value": f"{segment.time_numerator}/{segment.time_denominator}",
            },
            {"Property": "Bars", "Value": str(segment.bar_count)},
            {
                "Property": "RH token count",
                "Value": str(len(segment.right_hand_tokens)),
            },
            {"Property": "LH token count", "Value": str(len(segment.left_hand_tokens))},
            {
                "Property": "Difficulty level",
                "Value": str(segment.difficulty_level or "unlabeled"),
            },
        ],
        selection=None,
        label="Segment metadata",
    )
    meta_table
    return key_name, meta_table


@app.cell
def _(mo, segment):
    features = segment.difficulty_features

    feature_table = mo.ui.table(
        data=[
            {
                "Feature": "Max RH hand span (semitones)",
                "Value": str(features.max_right_hand_span_semitones),
            },
            {
                "Feature": "Max LH hand span (semitones)",
                "Value": str(features.max_left_hand_span_semitones),
            },
            {"Feature": "Notes per beat", "Value": f"{features.notes_per_beat:.2f}"},
            {
                "Feature": "Rhythmic diversity",
                "Value": f"{features.rhythmic_diversity:.2f}",
            },
            {
                "Feature": "Voice independence",
                "Value": f"{features.voice_independence:.2f}",
            },
            {"Feature": "Has accidentals", "Value": str(features.has_accidentals)},
            {"Feature": "Has dotted notes", "Value": str(features.has_dotted_notes)},
        ],
        selection=None,
        label="Difficulty features",
    )
    feature_table
    return (feature_table, features)


@app.cell
def _(mo, segments):
    from collections import Counter

    import altair as alt
    import pandas as pd

    from musak_model.tokens.schema import NoteToken as NT

    all_tokens = [token for seg in segments for token in seg.right_hand_tokens + seg.left_hand_tokens]

    duration_counts: Counter[str] = Counter()
    for _token in all_tokens:
        if isinstance(_token, NT):
            duration_counts[str(_token.duration_id)] += 1

    ordered_durations = sorted(duration_counts.keys(), key=int)
    rows = [{"duration": dur, "count": duration_counts.get(dur, 0)} for dur in ordered_durations]
    df = pd.DataFrame(rows)

    chart = mo.ui.altair_chart(
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("duration:N", sort=ordered_durations, title="Duration id"),
            y=alt.Y("count:Q", title="Token count"),
            tooltip=["duration", "count"],
        )
        .properties(title="Duration-id distribution across all segments", height=250)
    )
    chart
    return (
        Counter,
        NT,
        all_tokens,
        alt,
        chart,
        df,
        duration_counts,
        ordered_durations,
        pd,
        rows,
    )


if __name__ == "__main__":
    app.run()
