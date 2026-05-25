import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide", app_title="Dataset Statistics")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    from musak_model.decoder.notation import segment_to_score_data
    from musak_model.paths import DEFAULT_PROCESSED_ROOT
    from musak_model.processing.manifest import EncodedManifestField, ParsedManifestField
    from musak_shared.notation.html import score_data_html
    from notebooks.utils import (
        PitchSpelling,
        categorical_distribution,
        diagnostic_bucket_distribution,
        diagnostic_summary_rows,
        eligibility_distribution,
        encoded_run_directories,
        encoded_table_frame,
        hand_controls,
        ineligibility_reason_distribution,
        load_dataset_statistics,
        load_encoded_manifest_selection,
        overview_rows,
        parse_error_table_frame,
        parsed_table_frame,
        piano_roll_player_panel,
        processed_dataset_directories,
        reason_by_column,
        scale_root_distribution,
        segment_diagnostic_rows,
        segment_piano_roll_view_data,
        selected_table_row,
        table_records,
        token_histogram_distribution,
        token_summary_rows,
    )
    from notebooks.utils.statistics import COUNT_COLUMN, PERCENT_COLUMN, VALUE_COLUMN

    return (
        COUNT_COLUMN,
        DEFAULT_PROCESSED_ROOT,
        EncodedManifestField,
        PERCENT_COLUMN,
        ParsedManifestField,
        Path,
        PitchSpelling,
        VALUE_COLUMN,
        alt,
        categorical_distribution,
        diagnostic_bucket_distribution,
        diagnostic_summary_rows,
        eligibility_distribution,
        encoded_run_directories,
        encoded_table_frame,
        hand_controls,
        ineligibility_reason_distribution,
        scale_root_distribution,
        load_encoded_manifest_selection,
        load_dataset_statistics,
        mo,
        overview_rows,
        parse_error_table_frame,
        parsed_table_frame,
        piano_roll_player_panel,
        pd,
        processed_dataset_directories,
        reason_by_column,
        score_data_html,
        segment_diagnostic_rows,
        segment_piano_roll_view_data,
        segment_to_score_data,
        selected_table_row,
        table_records,
        token_histogram_distribution,
        token_summary_rows,
    )


@app.cell
def _(mo):
    title_output = mo.md("""
    # Dataset Statistics

    Processed dataset diagnostics from parsed and encoded manifests.
    """)
    title_output
    return


@app.cell
def _(DEFAULT_PROCESSED_ROOT, mo, processed_dataset_directories):
    dataset_directories = processed_dataset_directories(DEFAULT_PROCESSED_ROOT)
    dataset_options = {path.name: str(path) for path in dataset_directories}
    dataset_selector = mo.ui.dropdown(
        options=dataset_options,
        value=next(iter(dataset_options), None),
        label="Processed dataset",
        searchable=True,
    )
    dataset_selector
    return (dataset_selector,)


@app.cell
def _(Path, dataset_selector):
    dataset_dir = Path(dataset_selector.value) if dataset_selector.value is not None else None
    return (dataset_dir,)


@app.cell
def _(dataset_dir, encoded_run_directories, mo):
    if dataset_dir is None:
        encoded_selector = mo.ui.dropdown(options={}, label="Tokenizer run", searchable=True)
        encoded_selector_output = mo.md("")
    else:
        encoded_directorys = encoded_run_directories(dataset_dir)
        encoded_options = {path.name: str(path) for path in encoded_directorys}
        encoded_selector = mo.ui.dropdown(
            options=encoded_options,
            value=next(iter(encoded_options), None),
            label="Tokenizer run",
            searchable=True,
        )
        encoded_selector_output = (
            encoded_selector
            if encoded_options
            else mo.callout(
                "No encoded manifest found for this dataset. Segment-level sections will be hidden.",
                kind="warn",
            )
        )
    encoded_selector_output
    return (encoded_selector,)


@app.cell
def _(Path, encoded_selector):
    encoded_directory = Path(encoded_selector.value) if encoded_selector.value is not None else None
    return (encoded_directory,)


@app.cell
def _(mo):
    top_n_slider = mo.ui.slider(start=5, stop=50, step=1, value=20, label="Top categories")
    bpm_slider = mo.ui.slider(start=30, stop=240, step=1, value=60, label="BPM")
    prefer_flats_checkbox = mo.ui.checkbox(value=False, label="Prefer flats")
    controls_output = mo.hstack([top_n_slider, bpm_slider, prefer_flats_checkbox], justify="start", gap=2)
    controls_output
    return bpm_slider, prefer_flats_checkbox, top_n_slider


@app.cell
def _(dataset_dir, encoded_directory, load_dataset_statistics, pd):
    stats = None
    stats_error = ""
    if dataset_dir is not None:
        try:
            stats = load_dataset_statistics(dataset_dir, encoded_directory)
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as exception:
            stats_error = f"{type(exception).__name__}: {exception}"
    return stats, stats_error


@app.cell
def _(dataset_dir, encoded_directory, mo, stats_error):
    if dataset_dir is None:
        load_output = mo.callout("No processed dataset is available.", kind="warn")
    elif stats_error:
        load_output = mo.callout(stats_error, kind="danger")
    else:
        encoded_text = str(encoded_directory) if encoded_directory is not None else "none"
        load_output = mo.callout(
            f"Loaded dataset `{dataset_dir}` with encoded run `{encoded_text}`.",
            kind="success",
        )
    load_output
    return


@app.cell
def _(COUNT_COLUMN, PERCENT_COLUMN, VALUE_COLUMN, alt, mo):
    CHART_WIDTH = 360
    CHART_HEIGHT = 280
    WIDE_CHART_WIDTH = 760

    def horizontal_bar_chart(frame, *, title: str, value_title: str = "Category"):
        chart_frame = frame.copy()
        chart_frame["bar_color"] = chart_frame[VALUE_COLUMN].map(
            lambda value: "#2f855a" if str(value) == "no error" else "#4f7cac"
        )
        color = (
            alt.Color("bar_color:N", scale=None, legend=None)
            if "no error" in set(chart_frame[VALUE_COLUMN].astype(str))
            else alt.value("#4f7cac")
        )
        return (
            alt.Chart(chart_frame)
            .mark_bar()
            .encode(
                x=alt.X(f"{COUNT_COLUMN}:Q", title="Count"),
                y=alt.Y(f"{VALUE_COLUMN}:N", title=value_title, sort="-x"),
                color=color,
                tooltip=[
                    alt.Tooltip(f"{VALUE_COLUMN}:N", title=value_title),
                    alt.Tooltip(f"{COUNT_COLUMN}:Q", title="Count"),
                    alt.Tooltip(f"{PERCENT_COLUMN}:Q", title="Share", format=".1%"),
                ],
            )
            .properties(width=CHART_WIDTH, height=CHART_HEIGHT, title=title)
        )

    def token_distribution_chart(frame, *, eligible_column: str):
        tick_values = sorted(
            set(frame["token_bin_start"].astype(float)).union(set(frame["token_bin_end"].astype(float)))
        )
        return (
            alt.Chart(frame)
            .mark_bar()
            .encode(
                x=alt.X(
                    "token_bar_start:Q",
                    title="Token count",
                    axis=alt.Axis(values=tick_values, labelAngle=-45),
                    scale=alt.Scale(domain=[0, max(tick_values)], zero=True),
                ),
                x2="token_bar_end:Q",
                y=alt.Y(
                    f"{COUNT_COLUMN}:Q",
                    title="Segments",
                    stack=None,
                    scale=alt.Scale(zero=True),
                ),
                y2=alt.Y2(datum=0),
                order=alt.Order(f"{COUNT_COLUMN}:Q", sort="descending"),
                color=alt.Color(f"{eligible_column}:N", title="Eligible"),
                tooltip=[
                    alt.Tooltip("token_bin:N", title="Token count"),
                    alt.Tooltip(f"{COUNT_COLUMN}:Q", title="Segments"),
                    alt.Tooltip(f"{eligible_column}:N", title="Eligible"),
                ],
            )
            .properties(width=WIDE_CHART_WIDTH, height=320, title="Token count distribution")
        )

    def chart_output(chart):
        try:
            chart.to_dict()
        except Exception as exception:
            return mo.callout(f"Chart specification failed: {type(exception).__name__}: {exception}", kind="danger")

        return mo.ui.altair_chart(chart, chart_selection=False, legend_selection=False)

    def chart_grid(charts):
        columns = 3
        rows = []
        for start in range(0, len(charts), columns):
            rows.append(mo.hstack(charts[start : start + columns], justify="start", align="start", wrap=True, gap=2))
        return mo.vstack(rows, gap=2)

    return chart_grid, chart_output, horizontal_bar_chart, token_distribution_chart


@app.cell
def _(mo, overview_rows, stats):
    if stats is None:
        overview_output = mo.md("")
    else:
        overview_output = mo.ui.table(overview_rows(stats), selection=None, label="Overview")
    overview_output
    return


@app.cell
def _(
    ParsedManifestField,
    categorical_distribution,
    chart_grid,
    chart_output,
    horizontal_bar_chart,
    mo,
    stats,
    top_n_slider,
):
    if stats is None:
        parse_charts_output = mo.md("")
    else:
        status_chart = horizontal_bar_chart(
            categorical_distribution(stats.parsed, ParsedManifestField.STATUS, top_n=top_n_slider.value),
            title="Parse status",
            value_title="Status",
        )
        error_chart = horizontal_bar_chart(
            categorical_distribution(
                stats.parsed,
                ParsedManifestField.ERROR_TYPE,
                top_n=top_n_slider.value,
                empty_label="no error",
            ),
            title="Parse error types",
            value_title="Error type",
        )
        parse_charts_output = mo.vstack(
            [
                mo.md("## Parsing Quality"),
                chart_grid(
                    [
                        chart_output(status_chart),
                        chart_output(error_chart),
                    ],
                ),
            ],
            gap=2,
        )
    parse_charts_output
    return


@app.cell
def _(mo, parse_error_table_frame, stats):
    if stats is None:
        parse_error_output = mo.md("")
    else:
        parse_error_output = mo.ui.table(
            parse_error_table_frame(stats.parsed),
            selection=None,
            label="Parse errors",
        )
    parse_error_output
    return


@app.cell
def _(
    EncodedManifestField,
    chart_output,
    eligibility_distribution,
    horizontal_bar_chart,
    ineligibility_reason_distribution,
    mo,
    reason_by_column,
    stats,
    table_records,
):
    if stats is None or stats.encoded is None:
        eligibility_output = mo.callout("No encoded manifest is loaded.", kind="warn")
    else:
        eligibility_chart = horizontal_bar_chart(
            eligibility_distribution(stats.encoded),
            title="Segment eligibility",
            value_title="Eligibility",
        )
        reason_distribution = ineligibility_reason_distribution(stats.encoded)
        reason_chart = (
            mo.callout("No ineligibility reasons found.", kind="success")
            if reason_distribution.empty
            else mo.vstack(
                [
                    mo.md("### Ineligibility Reason Distribution"),
                    chart_output(
                        horizontal_bar_chart(
                            reason_distribution,
                            title="Ineligibility reasons",
                            value_title="Reason",
                        )
                    ),
                ],
                gap=1,
            )
        )
        reason_time_rows = table_records(reason_by_column(stats.encoded, EncodedManifestField.TIME_SIGNATURE))
        eligibility_output = mo.vstack(
            [
                mo.md("## Segment Eligibility"),
                mo.vstack(
                    [
                        mo.md("### Eligibility Distribution"),
                        chart_output(eligibility_chart),
                    ],
                    gap=1,
                ),
                reason_chart,
                mo.ui.table(reason_time_rows, selection=None, label="Reasons by time signature"),
            ],
            gap=2,
        )
    eligibility_output
    return


@app.cell
def _(
    EncodedManifestField,
    ParsedManifestField,
    categorical_distribution,
    chart_grid,
    chart_output,
    horizontal_bar_chart,
    mo,
    stats,
    scale_root_distribution,
    top_n_slider,
):
    if stats is None:
        music_output = mo.callout("No parsed or encoded manifest is loaded.", kind="warn")
    else:
        source = stats.encoded if stats.encoded is not None else stats.parsed
        key_column = (
            EncodedManifestField.SCALE_ROOT if stats.encoded is not None else ParsedManifestField.DECLARED_KEY_FIFTHS
        )
        time_column = (
            EncodedManifestField.TIME_SIGNATURE if stats.encoded is not None else ParsedManifestField.TIME_SIGNATURE
        )
        key_chart = horizontal_bar_chart(
            scale_root_distribution(source, key_column, top_n=top_n_slider.value),
            title="Scale root distribution" if stats.encoded is not None else "Declared fifths distribution",
            value_title="Scale root" if stats.encoded is not None else "Fifths",
        )
        scale_chart = (
            horizontal_bar_chart(
                categorical_distribution(source, EncodedManifestField.SCALE_TYPE, top_n=top_n_slider.value),
                title="Scale type distribution",
                value_title="Scale type",
            )
            if stats.encoded is not None
            else horizontal_bar_chart(
                categorical_distribution(source, ParsedManifestField.DECLARED_KEY_FIFTHS, top_n=top_n_slider.value),
                title="Declared fifths distribution",
                value_title="Fifths",
            )
        )
        time_chart = horizontal_bar_chart(
            categorical_distribution(source, time_column, top_n=top_n_slider.value),
            title="Time signature distribution",
            value_title="Time signature",
        )
        music_output = mo.vstack(
            [
                mo.md("## Musical Metadata"),
                chart_grid(
                    [
                        chart_output(key_chart),
                        chart_output(scale_chart),
                        chart_output(time_chart),
                    ],
                ),
            ],
            gap=2,
        )
    music_output
    return


@app.cell
def _(
    EncodedManifestField,
    chart_output,
    diagnostic_bucket_distribution,
    diagnostic_summary_rows,
    chart_grid,
    horizontal_bar_chart,
    mo,
    stats,
):
    if stats is None or stats.encoded is None:
        diagnostics_output = mo.callout("No encoded manifest is loaded.", kind="warn")
    else:
        diagnostic_charts = [
            horizontal_bar_chart(
                diagnostic_bucket_distribution(stats.encoded, EncodedManifestField.RIGHT_SILENCE_FRACTION),
                title="Right hand silence",
                value_title="Fraction",
            ),
            horizontal_bar_chart(
                diagnostic_bucket_distribution(stats.encoded, EncodedManifestField.LEFT_SILENCE_FRACTION),
                title="Left hand silence",
                value_title="Fraction",
            ),
            horizontal_bar_chart(
                diagnostic_bucket_distribution(stats.encoded, EncodedManifestField.BOTH_HANDS_SILENCE_FRACTION),
                title="Both hands silent",
                value_title="Fraction",
            ),
            horizontal_bar_chart(
                diagnostic_bucket_distribution(stats.encoded, EncodedManifestField.HAND_ACTIVITY_BALANCE),
                title="Hand activity balance",
                value_title="Fraction",
            ),
        ]
        diagnostics_output = mo.vstack(
            [
                mo.md("## Segment Diagnostics"),
                chart_grid([chart_output(chart) for chart in diagnostic_charts]),
                mo.ui.table(diagnostic_summary_rows(stats.encoded), selection=None, label="Diagnostic summary"),
            ],
            gap=2,
        )
    diagnostics_output
    return


@app.cell
def _(
    EncodedManifestField,
    chart_output,
    mo,
    stats,
    token_distribution_chart,
    token_histogram_distribution,
    token_summary_rows,
):
    if stats is None or stats.encoded is None:
        token_output = mo.callout("No encoded manifest is loaded.", kind="warn")
    else:
        token_chart = token_distribution_chart(
            token_histogram_distribution(stats.encoded),
            eligible_column=EncodedManifestField.ELIGIBLE_FOR_TRAINING,
        )
        token_output = mo.vstack(
            [
                mo.md("## Token Statistics"),
                chart_output(token_chart),
                mo.ui.table(token_summary_rows(stats.encoded), selection=None, label="Token summary"),
            ],
            gap=2,
        )
    token_output
    return


@app.cell
def _(EncodedManifestField, categorical_distribution, chart_output, horizontal_bar_chart, mo, stats, top_n_slider):
    if stats is None or stats.encoded is None:
        difficulty_output = mo.callout("No encoded manifest is loaded.", kind="warn")
    else:
        difficulty_chart = horizontal_bar_chart(
            categorical_distribution(
                stats.encoded,
                EncodedManifestField.DIFFICULTY_LEVEL,
                top_n=top_n_slider.value,
                empty_label="unlabeled",
            ),
            title="Difficulty labels",
            value_title="Difficulty",
        )
        difficulty_output = mo.vstack(
            [
                mo.md("## Difficulty"),
                chart_output(difficulty_chart),
            ],
            gap=2,
        )
    difficulty_output
    return


@app.cell
def _(encoded_table_frame, mo, parsed_table_frame, stats):
    if stats is None:
        table_output = mo.md("")
        encoded_manifest_table = None
    else:
        tables = [
            mo.ui.table(
                parsed_table_frame(stats.parsed),
                selection=None,
                label="Parsed manifest rows",
            )
        ]
        encoded_manifest_table = None
        if stats.encoded is not None:
            encoded_manifest_table = mo.ui.table(
                encoded_table_frame(stats.encoded),
                selection="single",
                label="Encoded manifest rows",
            )
            tables.append(encoded_manifest_table)
        table_output = mo.vstack([mo.md("## Manifest Rows"), *tables], gap=2)
    table_output
    return (encoded_manifest_table,)


@app.cell
def _(encoded_manifest_table, selected_table_row):
    selected_encoded_row = selected_table_row(encoded_manifest_table)
    return (selected_encoded_row,)


@app.cell
def _(dataset_dir, encoded_directory, load_encoded_manifest_selection, selected_encoded_row):
    selected_encoded_segment = None
    selected_encoded_error = ""
    selected_encoded_error_kind = "danger"
    if dataset_dir is not None and selected_encoded_row is not None:
        try:
            selected_encoded_segment = load_encoded_manifest_selection(
                selected_encoded_row,
                dataset_dir=dataset_dir,
                encoded_directory=encoded_directory,
            )
        except ValueError as exception:
            selected_encoded_error = f"{type(exception).__name__}: {exception}"
            selected_encoded_error_kind = "warn"
        except (FileNotFoundError, IndexError, TypeError) as exception:
            selected_encoded_error = f"{type(exception).__name__}: {exception}"
            selected_encoded_error_kind = "danger"

    return selected_encoded_error, selected_encoded_error_kind, selected_encoded_segment


@app.cell
def _(hand_controls, mo):
    selected_example_hand_controls = hand_controls(mo)
    return (selected_example_hand_controls,)


@app.cell
def _(
    PitchSpelling,
    alt,
    bpm_slider,
    mo,
    piano_roll_player_panel,
    prefer_flats_checkbox,
    score_data_html,
    segment_diagnostic_rows,
    segment_piano_roll_view_data,
    segment_to_score_data,
    selected_encoded_error,
    selected_encoded_error_kind,
    selected_encoded_row,
    selected_encoded_segment,
    selected_example_hand_controls,
):
    if selected_encoded_row is None:
        selected_example_output = mo.callout("Select an encoded manifest row to preview it.", kind="warn")
    elif selected_encoded_error:
        selected_example_output = mo.callout(selected_encoded_error, kind=selected_encoded_error_kind)
    elif selected_encoded_segment is None:
        selected_example_output = mo.md("")
    else:
        pitch_spelling = PitchSpelling.FLATS if prefer_flats_checkbox.value else PitchSpelling.SHARPS
        view_data = segment_piano_roll_view_data(
            selected_encoded_segment.segment,
            duration_vocabulary=selected_encoded_segment.duration_vocabulary,
            pitch_spelling=pitch_spelling,
            bpm=bpm_slider.value,
        )
        try:
            score_data = segment_to_score_data(
                selected_encoded_segment.segment,
                duration_vocabulary=selected_encoded_segment.duration_vocabulary,
                tempo=bpm_slider.value,
                measures_per_row=4,
            )
            iframe_height = f"{max(220, len(score_data.rows) * 140 + 24)}px"
            notation_output = mo.iframe(score_data_html(score_data), height=iframe_height)
        except ValueError as exception:
            notation_output = mo.callout(f"Notation rendering unavailable: {exception}", kind="warn")

        selected_example_output = mo.vstack(
            [
                mo.md("## Selected Encoded Example"),
                mo.ui.table([selected_encoded_row], selection=None, label="Selected manifest row"),
                notation_output,
                piano_roll_player_panel(
                    view_data,
                    mo=mo,
                    alt=alt,
                    bpm=bpm_slider.value,
                    controls=selected_example_hand_controls,
                ),
                mo.ui.table(
                    segment_diagnostic_rows(
                        selected_encoded_segment.segment,
                        duration_vocabulary=selected_encoded_segment.duration_vocabulary,
                    ),
                    selection=None,
                    label="Musical diagnostics",
                ),
            ],
            gap=2,
        )

    selected_example_output
    return


if __name__ == "__main__":
    app.run()
