from __future__ import annotations

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="wide", app_title="Synthetic Sight-Reading Generator")


@app.cell
def _():
    import altair as alt
    import marimo as mo

    from musak_model.decoder.notation import segment_to_score_data
    from musak_model.paths import DEFAULT_PROCESSED_ROOT
    from musak_model.tokens.schema import ScaleType
    from musak_shared.notation.html import score_data_html
    from notebooks.utils import (
        PitchSpelling,
        SyntheticGenerationRequest,
        baseline_overlay_chart,
        baseline_overlay_view_data,
        generate_synthetic_segment,
        hand_controls,
        load_synthetic_inputs,
        piano_roll_player_panel,
        segment_piano_roll_view_data,
        selected_directory,
    )

    alt.data_transformers.disable_max_rows()

    DEFAULT_GRID_COUNT_PER_BAR = 4
    return (
        DEFAULT_GRID_COUNT_PER_BAR,
        DEFAULT_PROCESSED_ROOT,
        PitchSpelling,
        ScaleType,
        SyntheticGenerationRequest,
        alt,
        baseline_overlay_chart,
        baseline_overlay_view_data,
        generate_synthetic_segment,
        hand_controls,
        load_synthetic_inputs,
        mo,
        piano_roll_player_panel,
        score_data_html,
        segment_piano_roll_view_data,
        segment_to_score_data,
        selected_directory,
    )


@app.cell
def _(mo):
    mo.md("# Synthetic Sight-Reading Generator")
    return


@app.cell
def _(DEFAULT_PROCESSED_ROOT, mo):
    figure_directory_browser = mo.ui.file_browser(
        initial_path=DEFAULT_PROCESSED_ROOT if DEFAULT_PROCESSED_ROOT.exists() else ".",
        selection_mode="directory",
        multiple=False,
        label="Figure artifact directory (containing counts.parquet and base_durations.parquet)",
    )
    setup_output = mo.vstack(
        [
            mo.md("## Setup"),
            mo.md(
                "Browse into a dataset's figure artifacts and select the directory holding "
                "`counts.parquet` and `base_durations.parquet` (e.g. `<encoded>/figure/all`)."
            ),
            figure_directory_browser,
        ],
        gap=2,
    )
    setup_output
    return (figure_directory_browser,)


@app.cell
def _(figure_directory_browser, load_synthetic_inputs, mo, selected_directory):
    synthetic_inputs = None
    if not figure_directory_browser.value:
        figure_status = mo.callout("Select a figure artifact directory produced by figure extraction.", kind="warn")
    else:
        directory_selection = selected_directory(figure_directory_browser, description="figure artifact")
        if directory_selection.path is None:
            figure_status = mo.callout(directory_selection.message or "Directory is unavailable.", kind="warn")
        else:
            try:
                with mo.status.spinner(title="Loading figure vocabulary..."):
                    synthetic_inputs = load_synthetic_inputs(directory_selection.path)
            except (FileNotFoundError, ValueError) as exception:
                figure_status = mo.callout(
                    f"Figure inputs are incomplete in `{directory_selection.path}`: {exception}",
                    kind="warn",
                )
            else:
                fitted = synthetic_inputs.fitted
                fit_summary = (
                    f"{len(fitted.register_overrides)} register + {len(fitted.accent_overrides)} accent overrides"
                    if fitted.register_overrides or fitted.accent_overrides
                    else "default parameters (no fitted_generator.json applied)"
                )
                figure_status = mo.callout(
                    f"Loaded `{directory_selection.path}`: "
                    f"{synthetic_inputs.figure_vocabulary.unique_count} figures, "
                    f"{synthetic_inputs.figure_vocabulary.total_count} occurrences. "
                    f"Fit: {fit_summary}.",
                    kind="success",
                )

    figure_status
    return (synthetic_inputs,)


@app.cell
def _(mo):
    generation_request, set_generation_request = mo.state(None)
    return generation_request, set_generation_request


@app.cell
def _(
    DEFAULT_GRID_COUNT_PER_BAR,
    ScaleType,
    SyntheticGenerationRequest,
    mo,
    set_generation_request,
    synthetic_inputs,
):
    mo.stop(synthetic_inputs is None, mo.md(""))

    scale_root = mo.ui.slider(start=0, stop=11, step=1, value=0, label="Scale root", show_value=True)
    scale_type = mo.ui.dropdown(
        options=[scale.value for scale in ScaleType], value=ScaleType.MAJOR.value, label="Scale"
    )
    time_numerator = mo.ui.number(start=1, stop=16, step=1, value=4, label="Time numerator")
    time_denominator = mo.ui.dropdown(options=["1", "2", "4", "8", "16"], value="4", label="Time denominator")
    grid_count_per_bar = mo.ui.number(
        start=1, stop=64, step=1, value=DEFAULT_GRID_COUNT_PER_BAR, label="Grid cells per bar"
    )
    chord_resolution = mo.ui.dropdown(options=["1", "2", "4", "8", "16"], value="1", label="Chord resolution")
    bar_count = mo.ui.number(start=1, stop=64, step=1, value=8, label="Bars")
    seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=1234, label="Seed")
    min_n = mo.ui.number(start=1, stop=8, step=1, value=2, label="Min figure length")
    max_n = mo.ui.number(start=1, stop=8, step=1, value=3, label="Max figure length")
    monophonic = mo.ui.checkbox(value=True, label="Monophonic")

    lambda_curve = mo.ui.slider(start=0.0, stop=5.0, step=0.05, value=1.0, label="λ curve", show_value=True)
    lambda_harm = mo.ui.slider(start=0.0, stop=5.0, step=0.05, value=1.0, label="λ harmony", show_value=True)
    lambda_accent = mo.ui.slider(start=0.0, stop=5.0, step=0.05, value=1.0, label="λ accent", show_value=True)
    commonness_bias = mo.ui.slider(start=0.0, stop=3.0, step=0.05, value=1.0, label="Commonness bias", show_value=True)
    max_resample_retries = mo.ui.number(start=1, stop=64, step=1, value=8, label="Max resample retries")

    arch_basis_count = mo.ui.number(start=1, stop=16, step=1, value=3, label="Arch basis count")
    arch_amplitude = mo.ui.slider(start=0.0, stop=10.0, step=0.1, value=4.0, label="Arch amplitude", show_value=True)
    arch_decay = mo.ui.slider(start=0.0, stop=3.0, step=0.05, value=1.0, label="Arch decay", show_value=True)
    ou_theta = mo.ui.slider(start=0.05, stop=1.0, step=0.05, value=0.2, label="OU theta", show_value=True)
    ou_sigma = mo.ui.slider(start=0.0, stop=3.0, step=0.05, value=1.0, label="OU sigma", show_value=True)

    baseline_logit = mo.ui.slider(start=-6.0, stop=6.0, step=0.1, value=-0.5, label="Baseline logit", show_value=True)
    metric_gain = mo.ui.slider(start=0.0, stop=10.0, step=0.1, value=2.0, label="Metric gain", show_value=True)
    metric_exponent = mo.ui.slider(start=0.0, stop=3.0, step=0.05, value=1.0, label="Metric exponent", show_value=True)
    envelope_basis_count = mo.ui.number(start=1, stop=16, step=1, value=3, label="Envelope basis count")
    envelope_amplitude = mo.ui.slider(
        start=0.0, stop=3.0, step=0.05, value=0.5, label="Envelope amplitude", show_value=True
    )
    envelope_decay = mo.ui.slider(start=0.0, stop=3.0, step=0.05, value=1.0, label="Envelope decay", show_value=True)

    co_activity_strength = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.7, label="Co-activity strength", show_value=True
    )
    activity_right = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.9, label="Right-hand activity", show_value=True
    )
    activity_left = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.9, label="Left-hand activity", show_value=True)
    sync_strength = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.0, label="Sync strength", show_value=True)
    self_transition_bias = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.25, label="Chord self-transition bias", show_value=True
    )
    functional_strength = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.7, label="Functional harmony strength", show_value=True
    )

    use_constraints = mo.ui.checkbox(value=True, label="Hard constraints")
    minimum_duration = mo.ui.dropdown(
        options=["None", "1/16", "1/8", "1/4", "1/2"], value="None", label="Shortest duration"
    )
    allow_dotted = mo.ui.checkbox(value=True, label="Allow dotted notes")
    max_notes_per_hand = mo.ui.number(start=0, stop=5, step=1, value=5, label="Max notes per hand (0 disables)")
    max_onset_span = mo.ui.number(start=0, stop=12, step=1, value=12, label="Max onset span (semitones)")
    max_gap = mo.ui.number(start=0, stop=36, step=1, value=0, label="Max melodic gap (0 disables)")
    max_span = mo.ui.number(start=0, stop=21, step=1, value=0, label="Static hand span (0 disables)")

    bpm = mo.ui.slider(start=30, stop=240, step=1, value=80, label="BPM", show_value=True)
    notation_bars = mo.ui.slider(start=1, stop=64, step=1, value=8, label="Notation bars", show_value=True)

    def _capture_request(_):
        set_generation_request(
            SyntheticGenerationRequest(
                scale_root=int(scale_root.value),
                scale_type=scale_type.value,
                time_numerator=int(time_numerator.value),
                time_denominator=int(time_denominator.value),
                grid_count_per_bar=int(grid_count_per_bar.value),
                chord_resolution=int(chord_resolution.value),
                bar_count=int(bar_count.value),
                seed=int(seed.value),
                min_n=int(min_n.value),
                max_n=int(max_n.value),
                monophonic=monophonic.value,
                lambda_curve=float(lambda_curve.value),
                lambda_harm=float(lambda_harm.value),
                lambda_accent=float(lambda_accent.value),
                commonness_bias=float(commonness_bias.value),
                max_resample_retries=int(max_resample_retries.value),
                arch_basis_count=int(arch_basis_count.value),
                arch_amplitude=float(arch_amplitude.value),
                arch_decay=float(arch_decay.value),
                ou_theta=float(ou_theta.value),
                ou_sigma=float(ou_sigma.value),
                baseline_logit=float(baseline_logit.value),
                metric_gain=float(metric_gain.value),
                metric_exponent=float(metric_exponent.value),
                envelope_basis_count=int(envelope_basis_count.value),
                envelope_amplitude=float(envelope_amplitude.value),
                envelope_decay=float(envelope_decay.value),
                co_activity_strength=float(co_activity_strength.value),
                activity_right=float(activity_right.value),
                activity_left=float(activity_left.value),
                sync_strength=float(sync_strength.value),
                self_transition_bias=float(self_transition_bias.value),
                functional_strength=float(functional_strength.value),
                use_constraints=use_constraints.value,
                minimum_duration=minimum_duration.value,
                allow_dotted=allow_dotted.value,
                max_notes_per_hand=int(max_notes_per_hand.value) or None,
                max_onset_span=int(max_onset_span.value) or None,
                max_gap=int(max_gap.value) or None,
                max_span=int(max_span.value) or None,
            )
        )

    generate_button = mo.ui.run_button(label="Generate", on_change=_capture_request)
    controls_output = mo.vstack(
        [
            mo.md("## Controls"),
            mo.md("### Musical Context"),
            mo.hstack(
                [
                    scale_root,
                    scale_type,
                    time_numerator,
                    time_denominator,
                    grid_count_per_bar,
                    chord_resolution,
                    bar_count,
                    seed,
                ],
                gap=2,
                wrap=True,
            ),
            mo.md("### Figures and Tilts"),
            mo.hstack(
                [min_n, max_n, monophonic, lambda_curve, lambda_harm, lambda_accent, commonness_bias],
                gap=2,
                wrap=True,
            ),
            mo.hstack([max_resample_retries], gap=2, wrap=True),
            mo.md("### Register Curve"),
            mo.hstack([arch_basis_count, arch_amplitude, arch_decay, ou_theta, ou_sigma], gap=2, wrap=True),
            mo.md("### Accent Field"),
            mo.hstack(
                [
                    baseline_logit,
                    metric_gain,
                    metric_exponent,
                    envelope_basis_count,
                    envelope_amplitude,
                    envelope_decay,
                ],
                gap=2,
                wrap=True,
            ),
            mo.md("### Hand Coupling and Harmony"),
            mo.hstack(
                [
                    co_activity_strength,
                    activity_right,
                    activity_left,
                    sync_strength,
                    self_transition_bias,
                    functional_strength,
                ],
                gap=2,
                wrap=True,
            ),
            mo.md("### Hard Constraints"),
            mo.hstack([use_constraints, minimum_duration, allow_dotted], gap=2, wrap=True),
            mo.hstack([max_notes_per_hand, max_onset_span, max_gap, max_span], gap=2, wrap=True),
            mo.md("### Playback and Display"),
            mo.hstack([bpm, notation_bars], gap=2, wrap=True),
            generate_button,
        ],
        gap=2,
    )
    controls_output
    return bpm, generate_button, notation_bars


@app.cell
def _(generate_synthetic_segment, generation_request, mo, synthetic_inputs):
    request = generation_request()
    if request is None or synthetic_inputs is None:
        output = None
    else:
        with mo.status.progress_bar(
            total=request.bar_count,
            title="Generating synthetic exercise...",
            remove_on_exit=True,
        ) as progress:

            def _update_progress(completed_bars, total_bars):
                progress.update(
                    title="Generating synthetic exercise...",
                    subtitle=f"bar {completed_bars}/{total_bars}",
                )

            output = generate_synthetic_segment(
                synthetic_inputs,
                request,
                progress_callback=_update_progress,
            )

    return (output,)


@app.cell
def _(hand_controls, mo):
    output_hand_controls = hand_controls(mo)
    return (output_hand_controls,)


@app.cell
def _(bpm, mo, notation_bars, output, score_data_html, segment_to_score_data):
    if output is None or output.segment is None:
        notation_output = mo.md("")
    elif output.decode_error is not None:
        notation_output = mo.callout(f"Notation skipped because decoding failed: {output.decode_error}", kind="warn")
    else:
        try:
            score_data = segment_to_score_data(
                output.segment,
                duration_vocabulary=output.duration_vocabulary,
                tempo=bpm.value,
                measures_per_row=4,
                max_bars=notation_bars.value,
                layout="grand_staff",
            )
            bar_note = (
                mo.callout(
                    f"Showing first {notation_bars.value} of {output.segment.bar_count} bar(s) in notation.",
                    kind="warn",
                )
                if output.segment.bar_count > notation_bars.value
                else mo.md("")
            )
            iframe_height = f"{max(260, len(score_data.rows) * 220 + 24)}px"
            notation_output = mo.vstack([bar_note, mo.iframe(score_data_html(score_data), height=iframe_height)], gap=1)
        except ValueError as exception:
            notation_output = mo.callout(f"Notation rendering unavailable: {exception}", kind="warn")

    notation_output
    return


@app.cell
def _(
    PitchSpelling,
    alt,
    bpm,
    mo,
    output,
    output_hand_controls,
    piano_roll_player_panel,
    segment_piano_roll_view_data,
):
    if output is None or output.segment is None:
        piano_roll_output = mo.md("")
    elif output.decode_error is not None:
        piano_roll_output = mo.callout(
            f"Piano roll skipped because decoding failed: {output.decode_error}",
            kind="warn",
        )
    else:
        view_data = segment_piano_roll_view_data(
            output.segment,
            duration_vocabulary=output.duration_vocabulary,
            pitch_spelling=PitchSpelling.SHARPS,
            bpm=bpm.value,
            title="Generated output piano roll",
        )
        piano_roll_output = piano_roll_player_panel(
            view_data,
            mo=mo,
            alt=alt,
            bpm=bpm.value,
            controls=output_hand_controls,
        )

    piano_roll_output
    return


@app.cell
def _(PitchSpelling, alt, baseline_overlay_chart, baseline_overlay_view_data, mo, output):
    if output is None or output.segment is None:
        baseline_output = mo.md("")
    else:
        baseline_view_data = baseline_overlay_view_data(output.trace, pitch_spelling=PitchSpelling.SHARPS)
        baseline_output = mo.ui.altair_chart(baseline_overlay_chart(baseline_view_data, alt=alt))

    baseline_output
    return


@app.cell
def _(mo, output):
    generation_status_output = (
        mo.md("") if output is None else mo.callout(output.status_message, kind=output.status_kind)
    )
    generation_status_output
    return


if __name__ == "__main__":
    app.run()
