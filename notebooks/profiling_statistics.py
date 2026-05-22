import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide", app_title="Profiling Statistics")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd

    from musak_model.paths import DEFAULT_PROCESSING_PROFILE_OUTPUT_DIR, DEFAULT_PROFILE_OUTPUT_DIR
    from notebooks.utils.profiling import (
        chart_frame,
        existing_directory,
        file_status_rows,
        has_columns,
        metric_rows,
        percentage_frame,
        profile_mode_paths,
        read_csv,
        read_json,
        selected_profile_root,
        sorted_frame,
    )

    return (
        DEFAULT_PROCESSING_PROFILE_OUTPUT_DIR,
        DEFAULT_PROFILE_OUTPUT_DIR,
        alt,
        chart_frame,
        existing_directory,
        file_status_rows,
        has_columns,
        metric_rows,
        mo,
        pd,
        percentage_frame,
        profile_mode_paths,
        read_csv,
        read_json,
        selected_profile_root,
        sorted_frame,
    )


@app.cell
def _(mo):
    title_output = mo.md("""
    # Profiling Statistics

    Processing profiler summaries from stage timings, CPU function profiling, source-level totals, and torch profiler
    reports.
    """)
    title_output
    return


@app.cell
def _(DEFAULT_PROCESSING_PROFILE_OUTPUT_DIR, DEFAULT_PROFILE_OUTPUT_DIR, existing_directory, mo):
    profile_root_browser = mo.ui.file_browser(
        initial_path=existing_directory(DEFAULT_PROCESSING_PROFILE_OUTPUT_DIR, fallback=DEFAULT_PROFILE_OUTPUT_DIR),
        selection_mode="directory",
        multiple=False,
        label="Profile root",
    )
    load_records_checkbox = mo.ui.checkbox(value=False, label="Load raw records")
    top_n_slider = mo.ui.slider(start=5, stop=100, step=1, value=25, label="Chart rows")
    controls_output = mo.vstack(
        [
            profile_root_browser,
            mo.hstack([top_n_slider, load_records_checkbox], gap=2, align="end"),
        ],
        gap=1,
    )
    controls_output
    return load_records_checkbox, profile_root_browser, top_n_slider


@app.cell
def _(DEFAULT_PROCESSING_PROFILE_OUTPUT_DIR, profile_root_browser, selected_profile_root):
    profile_root = selected_profile_root(profile_root_browser, default=DEFAULT_PROCESSING_PROFILE_OUTPUT_DIR)
    return (profile_root,)


@app.cell
def _(mo, profile_mode_paths, profile_root):
    profile_modes = profile_mode_paths(profile_root)
    profile_mode_selector = mo.ui.dropdown(
        options=list(profile_modes),
        value=next(iter(profile_modes), None),
        label="Mode",
        searchable=True,
    )
    profile_mode_output = (
        profile_mode_selector
        if profile_modes
        else mo.callout(
            f"No profile run artifacts found in `{profile_root}`.",
            kind="warn",
        )
    )
    profile_mode_output
    return profile_mode_selector, profile_modes


@app.cell
def _(profile_mode_selector, profile_modes, profile_root):
    profile_dir = profile_modes.get(profile_mode_selector.value, profile_root)
    return (profile_dir,)


@app.cell
def _(load_records_checkbox, pd, profile_dir, read_csv, read_json):
    summary = read_json(profile_dir / "summary.json")
    stage_stats = read_csv(profile_dir / "stage_stats.csv")
    source_stats = read_csv(profile_dir / "source_stats.csv")
    cpu_functions = read_csv(profile_dir / "cpu_profile_functions.csv")
    torch_functions = read_csv(profile_dir / "torch_profiler_functions.csv")
    records = read_csv(profile_dir / "records.csv") if load_records_checkbox.value else pd.DataFrame()
    return (
        cpu_functions,
        records,
        source_stats,
        stage_stats,
        summary,
        torch_functions,
    )


@app.cell
def _(file_status_rows, mo, profile_dir, pd):
    status_frame = pd.DataFrame(file_status_rows(profile_dir))
    existing_count = int(status_frame["exists"].sum()) if not status_frame.empty else 0
    if not profile_dir.exists():
        load_status = mo.callout(f"Profile directory does not exist: `{profile_dir}`", kind="warn")
    elif existing_count == 0:
        load_status = mo.callout(f"No profiler artifacts found in `{profile_dir}`.", kind="warn")
    else:
        load_status = mo.callout(f"Loaded {existing_count} profiler artifact(s) from `{profile_dir}`.", kind="success")

    mo.vstack([load_status, mo.ui.table(status_frame, selection=None, label="Artifact status")], gap=1)
    return


@app.cell
def _(alt, mo, pd):
    CHART_WIDTH = 520
    CHART_HEIGHT = 320

    def horizontal_bar_chart(frame, *, y_column: str, x_column: str, title: str, x_title: str):
        if frame.empty or y_column not in frame.columns or x_column not in frame.columns:
            return mo.callout(f"No `{x_column}` data available for {title}.", kind="warn")

        chart_frame = frame.copy()
        chart_frame[y_column] = chart_frame[y_column].astype(str)
        chart = (
            alt.Chart(chart_frame)
            .mark_bar()
            .encode(
                x=alt.X(f"{x_column}:Q", title=x_title),
                y=alt.Y(f"{y_column}:N", title="", sort="-x"),
                tooltip=[
                    alt.Tooltip(f"{y_column}:N", title=y_column),
                    alt.Tooltip(f"{x_column}:Q", title=x_title, format=".4f"),
                ],
            )
            .properties(width=CHART_WIDTH, height=CHART_HEIGHT, title=title)
        )
        return mo.ui.altair_chart(chart, chart_selection=False, legend_selection=False)

    return (horizontal_bar_chart,)


@app.cell
def _(cpu_functions, metric_rows, mo, pd, records, source_stats, stage_stats, summary, torch_functions):
    overview_frame = pd.DataFrame(
        metric_rows(summary, stage_stats, source_stats, cpu_functions, torch_functions, records)
    )
    mo.vstack(
        [
            mo.md("## Overview"),
            mo.ui.table(overview_frame, selection=None, label="Profile overview"),
        ],
        gap=1,
    )
    return (overview_frame,)


@app.cell
def _(
    chart_frame,
    has_columns,
    horizontal_bar_chart,
    mo,
    percentage_frame,
    sorted_frame,
    stage_stats,
    summary,
    top_n_slider,
):
    total_seconds = float(summary.get("total_seconds", 0.0) or 0.0)
    stage_table = sorted_frame(stage_stats)
    stage_table = percentage_frame(
        stage_table,
        value_column="total_seconds",
        total=total_seconds,
        output_column="profiled_time_share",
    )

    if not has_columns(stage_table, ("stage", "total_seconds", "mean_seconds")):
        stage_output = mo.md("")
    else:
        stage_chart_table = chart_frame(stage_table, row_count=top_n_slider.value)
        stage_output = mo.vstack(
            [
                mo.md("## Explicit Stage Timings"),
                mo.ui.table(stage_table, selection=None, label="Stage stats"),
                mo.hstack(
                    [
                        horizontal_bar_chart(
                            stage_chart_table,
                            y_column="stage",
                            x_column="total_seconds",
                            title="Total measured time by stage",
                            x_title="Seconds",
                        ),
                        horizontal_bar_chart(
                            stage_chart_table,
                            y_column="stage",
                            x_column="mean_seconds",
                            title="Mean time by stage",
                            x_title="Seconds",
                        ),
                    ],
                    gap=2,
                    align="start",
                    wrap=True,
                ),
            ],
            gap=2,
        )
    stage_output
    return (stage_table,)


@app.cell
def _(chart_frame, cpu_functions, has_columns, horizontal_bar_chart, mo, sorted_frame, top_n_slider):
    cpu_table = sorted_frame(cpu_functions)
    if not has_columns(cpu_table, ("function", "cumulative_seconds", "total_seconds")):
        cpu_output = mo.md("")
    else:
        cpu_chart_table = chart_frame(cpu_table, row_count=top_n_slider.value)
        cpu_output = mo.vstack(
            [
                mo.md("## CPU Function Profile"),
                mo.ui.table(cpu_table, selection=None, label="CPU functions"),
                mo.hstack(
                    [
                        horizontal_bar_chart(
                            cpu_chart_table,
                            y_column="function",
                            x_column="cumulative_seconds",
                            title="Cumulative CPU time by function",
                            x_title="Seconds",
                        ),
                        horizontal_bar_chart(
                            cpu_chart_table,
                            y_column="function",
                            x_column="total_seconds",
                            title="Self CPU time by function",
                            x_title="Seconds",
                        ),
                    ],
                    gap=2,
                    align="start",
                    wrap=True,
                ),
            ],
            gap=2,
        )
    cpu_output
    return (cpu_table,)


@app.cell
def _(chart_frame, has_columns, horizontal_bar_chart, mo, sorted_frame, torch_functions, top_n_slider):
    torch_table = sorted_frame(torch_functions)
    if not has_columns(torch_table, ("operation", "self_cpu_seconds", "self_cuda_seconds")):
        torch_output = mo.md("")
    else:
        torch_chart_table = chart_frame(torch_table, row_count=top_n_slider.value)
        torch_output = mo.vstack(
            [
                mo.md("## Torch Function Profile"),
                mo.ui.table(torch_table, selection=None, label="Torch operations"),
                mo.hstack(
                    [
                        horizontal_bar_chart(
                            torch_chart_table,
                            y_column="operation",
                            x_column="self_cuda_seconds",
                            title="Self CUDA time by operation",
                            x_title="Seconds",
                        ),
                        horizontal_bar_chart(
                            torch_chart_table,
                            y_column="operation",
                            x_column="self_cpu_seconds",
                            title="Self CPU time by operation",
                            x_title="Seconds",
                        ),
                    ],
                    gap=2,
                    align="start",
                    wrap=True,
                ),
            ],
            gap=2,
        )
    torch_output
    return (torch_table,)


@app.cell
def _(chart_frame, has_columns, horizontal_bar_chart, mo, sorted_frame, source_stats, top_n_slider):
    source_table = sorted_frame(source_stats)
    if not has_columns(source_table, ("source_file", "total_seconds")):
        source_output = mo.md("")
    else:
        source_chart_table = chart_frame(source_table, row_count=top_n_slider.value)
        source_output = mo.vstack(
            [
                mo.md("## Source Files"),
                mo.ui.table(source_table, selection=None, label="Slowest sources"),
                horizontal_bar_chart(
                    source_chart_table,
                    y_column="source_file",
                    x_column="total_seconds",
                    title="Total measured time by source",
                    x_title="Seconds",
                ),
            ],
            gap=2,
        )
    source_output
    return (source_table,)


@app.cell
def _(mo, records, sorted_frame):
    if records.empty:
        records_output = mo.md("")
    else:
        records_table = sorted_frame(records, sort_column="seconds")
        records_output = mo.vstack(
            [
                mo.md("## Raw Records"),
                mo.ui.table(records_table, selection=None, label="Raw timing records"),
            ],
            gap=1,
        )

    records_output
    return


if __name__ == "__main__":
    app.run()
