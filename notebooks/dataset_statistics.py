import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide", app_title="Dataset Statistics")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd

    from musak_model.paths import DEFAULT_PROCESSED_ROOT
    from musak_model.processing.manifest import EncodedManifestField, ParsedManifestField
    from notebooks.utils import (
        categorical_distribution,
        eligibility_distribution,
        encoded_run_dirs,
        encoded_table_rows,
        ineligibility_reason_distribution,
        load_dataset_statistics,
        overview_rows,
        parsed_table_rows,
        processed_dataset_dirs,
        reason_by_column,
        token_summary_rows,
        top_parse_error_rows,
    )
    from notebooks.utils.statistics import COUNT_COLUMN, PERCENT_COLUMN, VALUE_COLUMN

    return (
        COUNT_COLUMN,
        DEFAULT_PROCESSED_ROOT,
        EncodedManifestField,
        PERCENT_COLUMN,
        ParsedManifestField,
        Path,
        VALUE_COLUMN,
        alt,
        categorical_distribution,
        eligibility_distribution,
        encoded_run_dirs,
        encoded_table_rows,
        ineligibility_reason_distribution,
        load_dataset_statistics,
        mo,
        overview_rows,
        parsed_table_rows,
        pd,
        processed_dataset_dirs,
        reason_by_column,
        token_summary_rows,
        top_parse_error_rows,
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
def _(DEFAULT_PROCESSED_ROOT, mo, processed_dataset_dirs):
    dataset_dirs = processed_dataset_dirs(DEFAULT_PROCESSED_ROOT)
    dataset_options = {path.name: str(path) for path in dataset_dirs}
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
def _(dataset_dir, encoded_run_dirs, mo):
    if dataset_dir is None:
        encoded_selector = mo.ui.dropdown(options={}, label="Tokenizer run", searchable=True)
        encoded_selector_output = mo.md("")
    else:
        encoded_dirs = encoded_run_dirs(dataset_dir)
        encoded_options = {path.name: str(path) for path in encoded_dirs}
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
    encoded_dir = Path(encoded_selector.value) if encoded_selector.value is not None else None
    return (encoded_dir,)


@app.cell
def _(mo):
    top_n_slider = mo.ui.slider(start=5, stop=50, step=1, value=20, label="Top categories")
    table_limit_slider = mo.ui.slider(start=10, stop=500, step=10, value=50, label="Table rows")
    controls_output = mo.hstack([top_n_slider, table_limit_slider], gap=2)
    controls_output
    return table_limit_slider, top_n_slider


@app.cell
def _(dataset_dir, encoded_dir, load_dataset_statistics, pd):
    stats = None
    stats_error = ""
    if dataset_dir is not None:
        try:
            stats = load_dataset_statistics(dataset_dir, encoded_dir)
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as exception:
            stats_error = f"{type(exception).__name__}: {exception}"
    return stats, stats_error


@app.cell
def _(dataset_dir, encoded_dir, mo, stats_error):
    if dataset_dir is None:
        load_output = mo.callout("No processed dataset is available.", kind="warn")
    elif stats_error:
        load_output = mo.callout(stats_error, kind="danger")
    else:
        encoded_text = str(encoded_dir) if encoded_dir is not None else "none"
        load_output = mo.callout(
            f"Loaded dataset `{dataset_dir}` with encoded run `{encoded_text}`.",
            kind="success",
        )
    load_output
    return


@app.cell
def _(COUNT_COLUMN, PERCENT_COLUMN, VALUE_COLUMN, alt):
    def horizontal_bar_chart(frame, *, title: str, value_title: str = "Category"):
        return (
            alt.Chart(frame)
            .mark_bar()
            .encode(
                x=alt.X(f"{COUNT_COLUMN}:Q", title="Count"),
                y=alt.Y(f"{VALUE_COLUMN}:N", title=value_title, sort="-x"),
                tooltip=[
                    alt.Tooltip(f"{VALUE_COLUMN}:N", title=value_title),
                    alt.Tooltip(f"{COUNT_COLUMN}:Q", title="Count"),
                    alt.Tooltip(f"{PERCENT_COLUMN}:Q", title="Share", format=".1%"),
                ],
            )
            .properties(width="container", height=320, title=title)
        )

    def token_histogram(frame, *, token_column: str, eligible_column: str):
        return (
            alt.Chart(frame)
            .mark_bar()
            .encode(
                x=alt.X(f"{token_column}:Q", bin=alt.Bin(maxbins=40), title="Token count"),
                y=alt.Y("count():Q", title="Segments"),
                color=alt.Color(f"{eligible_column}:N", title="Eligible"),
                tooltip=[
                    alt.Tooltip(f"{token_column}:Q", title="Token count"),
                    alt.Tooltip("count():Q", title="Segments"),
                    alt.Tooltip(f"{eligible_column}:N", title="Eligible"),
                ],
            )
            .properties(width="container", height=320, title="Token count distribution")
        )

    return horizontal_bar_chart, token_histogram


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
        diagnostics_chart = horizontal_bar_chart(
            categorical_distribution(stats.parsed, "has_parse_diagnostics", top_n=top_n_slider.value),
            title="Parse diagnostics present",
            value_title="Diagnostics",
        )
        parse_charts_output = mo.vstack(
            [
                mo.md("## Parsing Quality"),
                mo.hstack(
                    [
                        mo.ui.altair_chart(status_chart),
                        mo.ui.altair_chart(error_chart),
                        mo.ui.altair_chart(diagnostics_chart),
                    ],
                    gap=2,
                ),
            ],
            gap=2,
        )
    parse_charts_output
    return


@app.cell
def _(mo, stats, table_limit_slider, top_parse_error_rows):
    if stats is None:
        parse_error_output = mo.md("")
    else:
        parse_error_output = mo.ui.table(
            top_parse_error_rows(stats.parsed, limit=table_limit_slider.value),
            selection=None,
            label="Parse errors",
        )
    parse_error_output
    return


@app.cell
def _(
    EncodedManifestField,
    eligibility_distribution,
    horizontal_bar_chart,
    ineligibility_reason_distribution,
    mo,
    reason_by_column,
    stats,
):
    if stats is None or stats.encoded is None:
        eligibility_output = mo.md("")
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
            else mo.ui.altair_chart(
                horizontal_bar_chart(
                    reason_distribution,
                    title="Ineligibility reasons",
                    value_title="Reason",
                )
            )
        )
        reason_time_rows = reason_by_column(stats.encoded, EncodedManifestField.TIME_SIGNATURE).to_dict("records")
        eligibility_output = mo.vstack(
            [
                mo.md("## Segment Eligibility"),
                mo.hstack([mo.ui.altair_chart(eligibility_chart), reason_chart], gap=2),
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
    horizontal_bar_chart,
    mo,
    stats,
    top_n_slider,
):
    if stats is None:
        music_output = mo.md("")
    else:
        source = stats.encoded if stats.encoded is not None else stats.parsed
        key_column = EncodedManifestField.KEY_ROOT if stats.encoded is not None else ParsedManifestField.KEY_ROOT
        scale_column = EncodedManifestField.SCALE_TYPE if stats.encoded is not None else ParsedManifestField.SCALE_TYPE
        time_column = (
            EncodedManifestField.TIME_SIGNATURE if stats.encoded is not None else ParsedManifestField.TIME_SIGNATURE
        )
        key_chart = horizontal_bar_chart(
            categorical_distribution(source, key_column, top_n=top_n_slider.value),
            title="Key root distribution",
            value_title="Key root",
        )
        scale_chart = horizontal_bar_chart(
            categorical_distribution(source, scale_column, top_n=top_n_slider.value),
            title="Scale type distribution",
            value_title="Scale type",
        )
        time_chart = horizontal_bar_chart(
            categorical_distribution(source, time_column, top_n=top_n_slider.value),
            title="Time signature distribution",
            value_title="Time signature",
        )
        music_output = mo.vstack(
            [
                mo.md("## Musical Metadata"),
                mo.hstack(
                    [
                        mo.ui.altair_chart(key_chart),
                        mo.ui.altair_chart(scale_chart),
                        mo.ui.altair_chart(time_chart),
                    ],
                    gap=2,
                ),
            ],
            gap=2,
        )
    music_output
    return


@app.cell
def _(EncodedManifestField, mo, stats, token_histogram, token_summary_rows):
    if stats is None or stats.encoded is None:
        token_output = mo.md("")
    else:
        token_chart = token_histogram(
            stats.encoded,
            token_column=EncodedManifestField.TOKEN_COUNT,
            eligible_column=EncodedManifestField.ELIGIBLE_FOR_TRAINING,
        )
        token_output = mo.vstack(
            [
                mo.md("## Token Statistics"),
                mo.hstack(
                    [
                        mo.ui.altair_chart(token_chart),
                        mo.ui.table(token_summary_rows(stats.encoded), selection=None, label="Token summary"),
                    ],
                    gap=2,
                ),
            ],
            gap=2,
        )
    token_output
    return


@app.cell
def _(EncodedManifestField, categorical_distribution, horizontal_bar_chart, mo, stats, top_n_slider):
    if stats is None or stats.encoded is None:
        difficulty_output = mo.md("")
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
                mo.ui.altair_chart(difficulty_chart),
            ],
            gap=2,
        )
    difficulty_output
    return


@app.cell
def _(encoded_table_rows, mo, parsed_table_rows, stats, table_limit_slider):
    if stats is None:
        table_output = mo.md("")
    else:
        tables = [
            mo.ui.table(
                parsed_table_rows(stats.parsed, limit=table_limit_slider.value),
                selection=None,
                label="Parsed manifest rows",
            )
        ]
        if stats.encoded is not None:
            tables.append(
                mo.ui.table(
                    encoded_table_rows(stats.encoded, limit=table_limit_slider.value),
                    selection=None,
                    label="Encoded manifest rows",
                )
            )
        table_output = mo.vstack([mo.md("## Manifest Rows"), *tables], gap=2)
    table_output
    return


if __name__ == "__main__":
    app.run()
