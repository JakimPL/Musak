from __future__ import annotations

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide", app_title="N-Gram Analysis")


@app.cell
def _():
    from fractions import Fraction
    from pathlib import Path
    from typing import Final

    import altair as alt
    import marimo as mo
    import pandas as pd

    from musak_model.n_grams.profile.io import COUNT_COLUMN, HAND_COLUMN, N_COLUMN, SCALE_TYPE_COLUMN
    from musak_model.paths import DEFAULT_ANALYSIS_DIR
    from musak_shared.notation.html import score_data_html
    from notebooks.utils import (
        FIGURE_LABEL_COLUMN,
        FIGURE_PERCENT_COLUMN,
        FIGURE_PROPERTY_COLUMN,
        FIGURE_PROPERTY_VALUE_COLUMN,
        FIGURE_TEXT_COLUMN,
        analysis_result_files,
        figure_display_unit,
        figure_filter_frame,
        figure_group_summary,
        figure_ngram_to_score_data,
        figure_property_distribution,
        parse_figure_ngram,
        read_figure_count_frame,
        selected_table_row,
        table_records,
        top_figure_frame,
    )

    alt.data_transformers.disable_max_rows()
    n_all_option: Final[str] = "all"
    return (
        COUNT_COLUMN,
        DEFAULT_ANALYSIS_DIR,
        FIGURE_LABEL_COLUMN,
        FIGURE_PERCENT_COLUMN,
        FIGURE_PROPERTY_COLUMN,
        FIGURE_PROPERTY_VALUE_COLUMN,
        FIGURE_TEXT_COLUMN,
        Fraction,
        HAND_COLUMN,
        N_COLUMN,
        n_all_option,
        Path,
        SCALE_TYPE_COLUMN,
        alt,
        analysis_result_files,
        figure_display_unit,
        figure_filter_frame,
        figure_group_summary,
        figure_ngram_to_score_data,
        figure_property_distribution,
        mo,
        parse_figure_ngram,
        pd,
        read_figure_count_frame,
        score_data_html,
        selected_table_row,
        table_records,
        top_figure_frame,
    )


@app.cell
def _(mo):
    mo.md("""
    # N-Gram Analysis

    Figure n-gram counts extracted from encoded datasets.
    """)
    return


@app.cell
def _(DEFAULT_ANALYSIS_DIR, analysis_result_files, mo):
    result_files = analysis_result_files(DEFAULT_ANALYSIS_DIR)
    result_options = {path.name: str(path) for path in result_files}
    result_selector = mo.ui.dropdown(
        options=result_options,
        value=next(iter(result_options), None),
        label="Analysis result",
        searchable=True,
    )
    result_selector if result_options else mo.callout("No analysis CSV files found.", kind="warn")
    return (result_selector,)


@app.cell
def _(Path, result_selector):
    result_path = Path(result_selector.value) if result_selector.value is not None else None
    return (result_path,)


@app.cell
def _(mo):
    top_n = mo.ui.slider(start=5, stop=100, step=1, value=20, label="Top figures")
    preferred_unit = mo.ui.dropdown(
        options=["1/4", "1/8", "1/16", "1/32"],
        value="1/8",
        label="Display unit",
    )
    controls = mo.hstack([top_n, preferred_unit], gap=2, justify="start")
    controls
    return preferred_unit, top_n


@app.cell
def _(mo, pd, read_figure_count_frame, result_path):
    frame = None
    load_error = ""
    if result_path is not None:
        try:
            frame = read_figure_count_frame(result_path)
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as exception:
            load_error = f"{type(exception).__name__}: {exception}"

    if result_path is None:
        load_output = mo.callout("Select an analysis result CSV.", kind="warn")
    elif load_error:
        load_output = mo.callout(load_error, kind="danger")
    else:
        load_output = mo.callout(f"Loaded `{result_path}`.", kind="success")

    load_output
    return frame, load_error


@app.cell
def _(HAND_COLUMN, N_COLUMN, SCALE_TYPE_COLUMN, frame, mo, n_all_option):
    if frame is None or frame.empty:
        scale_selector = mo.ui.dropdown(options={}, label="Scale")
        hand_selector = mo.ui.dropdown(options={}, label="Hand")
        n_selector = mo.ui.dropdown(options={}, label="n")
        filter_output = mo.md("")
    else:
        scale_values = sorted(str(value) for value in frame[SCALE_TYPE_COLUMN].dropna().unique())
        hand_values = sorted(str(value) for value in frame[HAND_COLUMN].dropna().unique())
        n_values = sorted(int(value) for value in frame[N_COLUMN].dropna().unique())
        scale_selector = mo.ui.dropdown(options=scale_values, value=scale_values[0], label="Scale")
        hand_selector = mo.ui.dropdown(options=hand_values, value=hand_values[0], label="Hand")
        n_options = [n_all_option, *(str(value) for value in n_values)]
        n_selector = mo.ui.dropdown(
            options=n_options,
            value=n_all_option,
            label="n",
        )
        filter_output = mo.hstack([scale_selector, hand_selector, n_selector], gap=2, justify="start")

    filter_output
    return hand_selector, n_selector, scale_selector


@app.cell
def _(n_all_option, n_selector):
    selected_n = None if n_selector.value in (None, n_all_option) else int(n_selector.value)
    return (selected_n,)


@app.cell
def _(COUNT_COLUMN, HAND_COLUMN, N_COLUMN, SCALE_TYPE_COLUMN, alt, figure_group_summary, frame, mo):
    if frame is None or frame.empty:
        summary_output = mo.md("")
    else:
        summary = figure_group_summary(frame)
        summary_chart = (
            alt.Chart(summary)
            .mark_bar()
            .encode(
                x=alt.X(f"{N_COLUMN}:O", title="n"),
                y=alt.Y("total_count:Q", title="Occurrences"),
                color=alt.Color(f"{HAND_COLUMN}:N", title="Hand"),
                column=alt.Column(f"{SCALE_TYPE_COLUMN}:N", title="Scale"),
                tooltip=[
                    alt.Tooltip(f"{SCALE_TYPE_COLUMN}:N", title="Scale"),
                    alt.Tooltip(f"{HAND_COLUMN}:N", title="Hand"),
                    alt.Tooltip(f"{N_COLUMN}:O", title="n"),
                    alt.Tooltip("total_count:Q", title="Occurrences"),
                    alt.Tooltip("unique_figures:Q", title="Unique figures"),
                ],
            )
            .properties(width=140, height=260, title="Figure counts by group")
        )
        summary_output = mo.vstack(
            [
                mo.ui.altair_chart(summary_chart, chart_selection=False, legend_selection=False),
                mo.ui.table(summary, selection=None),
            ],
            gap=2,
        )

    summary_output
    return


@app.cell
def _(
    COUNT_COLUMN,
    FIGURE_PERCENT_COLUMN,
    FIGURE_PROPERTY_COLUMN,
    FIGURE_PROPERTY_VALUE_COLUMN,
    FIGURE_TEXT_COLUMN,
    alt,
    figure_filter_frame,
    figure_property_distribution,
    frame,
    hand_selector,
    mo,
    scale_selector,
    selected_n,
    top_figure_frame,
    top_n,
):
    if frame is None or frame.empty:
        filtered_frame = None
        top_frame = None
        top_output = mo.md("")
        top_table = mo.ui.table([])
    else:
        filtered_frame = figure_filter_frame(
            frame,
            scale_type=scale_selector.value,
            hand=hand_selector.value,
            n=selected_n,
        )
        top_frame = top_figure_frame(
            frame,
            scale_type=scale_selector.value,
            hand=hand_selector.value,
            n=selected_n,
            top_n=int(top_n.value),
        )
        property_distribution = figure_property_distribution(filtered_frame)
        property_chart = (
            alt.Chart(property_distribution)
            .mark_bar()
            .encode(
                x=alt.X(f"{FIGURE_PROPERTY_COLUMN}:N", title="Property"),
                y=alt.Y(f"{FIGURE_PERCENT_COLUMN}:Q", title="Share", axis=alt.Axis(format="%")),
                color=alt.Color(f"{FIGURE_PROPERTY_VALUE_COLUMN}:N", title="Value"),
                tooltip=[
                    alt.Tooltip(f"{FIGURE_PROPERTY_COLUMN}:N", title="Property"),
                    alt.Tooltip(f"{FIGURE_PROPERTY_VALUE_COLUMN}:N", title="Value"),
                    alt.Tooltip(f"{COUNT_COLUMN}:Q", title="Count"),
                    alt.Tooltip(f"{FIGURE_PERCENT_COLUMN}:Q", title="Share", format=".1%"),
                ],
            )
            .properties(width=360, height=220, title="Figure property distribution")
        )
        top_chart = (
            alt.Chart(top_frame)
            .mark_bar()
            .encode(
                x=alt.X(f"{COUNT_COLUMN}:Q", title="Count"),
                y=alt.Y(
                    f"{FIGURE_TEXT_COLUMN}:N",
                    title="Figure",
                    sort=None,
                    axis=alt.Axis(labelLimit=620),
                ),
                tooltip=[
                    alt.Tooltip(f"{FIGURE_TEXT_COLUMN}:N", title="Figure"),
                    alt.Tooltip(f"{COUNT_COLUMN}:Q", title="Count"),
                    alt.Tooltip(f"{FIGURE_PERCENT_COLUMN}:Q", title="Share", format=".1%"),
                ],
            )
            .properties(width=680, height=max(180, 24 * len(top_frame)), title="Most common figures")
        )
        table_frame = filtered_frame.drop(columns=["figure"])
        top_table = mo.ui.table(table_frame, selection="single", page_size=min(max(len(table_frame), 1), 20))
        top_output = mo.vstack(
            [
                mo.ui.altair_chart(property_chart, chart_selection=False, legend_selection=False),
                mo.ui.altair_chart(top_chart, chart_selection=False, legend_selection=False),
                top_table,
            ],
            gap=2,
        )

    top_output
    return filtered_frame, top_frame, top_table


@app.cell
def _(
    FIGURE_LABEL_COLUMN,
    Fraction,
    filtered_frame,
    figure_display_unit,
    figure_ngram_to_score_data,
    mo,
    parse_figure_ngram,
    preferred_unit,
    score_data_html,
    selected_table_row,
    top_table,
):
    row = selected_table_row(top_table)
    if row is None or filtered_frame is None:
        notation_output = mo.md("")
    else:
        try:
            figure_index = int(str(row[FIGURE_LABEL_COLUMN]).removeprefix("#")) - 1
            figure = parse_figure_ngram(str(filtered_frame.iloc[figure_index]["figure"]))
            unit = Fraction(str(preferred_unit.value))
            resolved_unit = figure_display_unit(figure, preferred_unit=unit)
            score_data = figure_ngram_to_score_data(figure, preferred_unit=unit)
            notation_output = mo.vstack(
                [
                    mo.md(f"Display unit: `{resolved_unit}`"),
                    mo.iframe(score_data_html(score_data, element_id="n-gram-notation"), height=170),
                ],
                gap=1,
            )
        except (KeyError, ValueError) as exception:
            notation_output = mo.callout(f"Notation rendering unavailable: {exception}", kind="warn")

    notation_output
    return


if __name__ == "__main__":
    app.run()
