from __future__ import annotations

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="wide", app_title="Form-Driven Surface Renderer")


@app.cell
def _():
    import altair as alt
    import marimo as mo

    from musak_model.decoder.notation import segment_to_score_data
    from musak_model.paths import DEFAULT_PROCESSED_ROOT
    from musak_model.synthetic.fitting.form.fit import FormFittingConfig
    from musak_model.tokens.schema import ScaleType
    from musak_shared.notation.html import score_data_html
    from notebooks.utils import (
        FormRenderRequest,
        PitchSpelling,
        hand_controls,
        load_synthetic_inputs,
        piano_roll_player_panel,
        render_form_segment,
        scale_pitch_class_set,
        segment_piano_roll_view_data,
        selected_directory,
    )

    alt.data_transformers.disable_max_rows()
    return (
        DEFAULT_PROCESSED_ROOT,
        FormFittingConfig,
        FormRenderRequest,
        PitchSpelling,
        ScaleType,
        alt,
        hand_controls,
        load_synthetic_inputs,
        mo,
        piano_roll_player_panel,
        render_form_segment,
        scale_pitch_class_set,
        score_data_html,
        segment_piano_roll_view_data,
        segment_to_score_data,
        selected_directory,
    )


@app.cell
def _(mo):
    mo.md(
        "# Form-Driven Surface Renderer\n"
        "Top-down path (Phase 0–3): a `FormPrior` samples a `FormTree` (phrases + per-phrase cadences "
        "+ repetition classes), the harmony grammar roots per phrase, and the surface renderer fills the "
        "metrical slots. Pick **fitted** to drive it from the corpus-learned prior, or **fallback** for the "
        "YAML prior (works with no corpus fit)."
    )
    return


@app.cell
def _(DEFAULT_PROCESSED_ROOT, mo):
    figure_directory_browser = mo.ui.file_browser(
        initial_path=DEFAULT_PROCESSED_ROOT if DEFAULT_PROCESSED_ROOT.exists() else ".",
        selection_mode="directory",
        multiple=False,
        label="Figure artifact directory (the `<encoded>/figure` root, holding all/ and form/)",
    )
    mo.vstack(
        [
            mo.md("## Setup"),
            mo.md(
                "Select a dataset's figure artifact root produced by figure extraction (and, for the fitted "
                "prior, `extract_form_statistics` + `make fit-generator`)."
            ),
            figure_directory_browser,
        ],
        gap=2,
    )
    return (figure_directory_browser,)


@app.cell
def _(FormFittingConfig, figure_directory_browser, load_synthetic_inputs, mo, selected_directory):
    synthetic_inputs = None
    form_fitting = FormFittingConfig.load()
    if not figure_directory_browser.value:
        setup_status = mo.callout("Select a figure artifact directory produced by figure extraction.", kind="warn")
    else:
        directory_selection = selected_directory(figure_directory_browser, description="figure artifact")
        if directory_selection.path is None:
            setup_status = mo.callout(directory_selection.message or "Directory is unavailable.", kind="warn")
        else:
            try:
                with mo.status.spinner(title="Loading figure vocabulary and fitted config..."):
                    synthetic_inputs = load_synthetic_inputs(directory_selection.path)
            except (FileNotFoundError, ValueError) as exception:
                setup_status = mo.callout(f"Figure inputs are incomplete: {exception}", kind="warn")
            else:
                fitted_scales = sorted(scale.value for scale in synthetic_inputs.fitted.form_priors)
                fit_summary = (
                    f"fitted form priors: {fitted_scales}" if fitted_scales else "no fitted form priors (use fallback)"
                )
                setup_status = mo.callout(
                    f"Loaded `{directory_selection.path}`: "
                    f"{synthetic_inputs.figure_vocabulary.unique_count} figures; {fit_summary}.",
                    kind="success",
                )

    setup_status
    return form_fitting, synthetic_inputs


@app.cell
def _(mo):
    render_request, set_render_request = mo.state(None)
    return render_request, set_render_request


@app.cell
def _(FormRenderRequest, ScaleType, mo, set_render_request, synthetic_inputs):
    mo.stop(synthetic_inputs is None, mo.md(""))

    scale_root = mo.ui.slider(start=0, stop=11, step=1, value=0, label="Scale root", show_value=True)
    scale_type = mo.ui.dropdown(
        options=[scale.value for scale in ScaleType], value=ScaleType.MAJOR.value, label="Scale"
    )
    time_numerator = mo.ui.number(start=1, stop=16, step=1, value=4, label="Time numerator")
    time_denominator = mo.ui.dropdown(options=["1", "2", "4", "8", "16"], value="4", label="Time denominator")
    bar_count = mo.ui.number(start=2, stop=32, step=1, value=8, label="Bars")
    seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=0, label="Seed")
    harmonic_slot_denominator = mo.ui.dropdown(options=["1", "2", "4"], value="1", label="Harmonic slot (1/N note)")
    prior_source = mo.ui.dropdown(options=["fitted", "fallback"], value="fitted", label="Form prior")
    commonness_bias = mo.ui.slider(start=0.0, stop=3.0, step=0.1, value=1.0, label="Commonness bias", show_value=True)
    lambda_curve = mo.ui.slider(start=0.0, stop=8.0, step=0.5, value=2.0, label="λ curve", show_value=True)
    lambda_harmonic = mo.ui.slider(start=0.0, stop=8.0, step=0.5, value=4.0, label="λ harmonic", show_value=True)
    lambda_accent = mo.ui.slider(start=0.0, stop=4.0, step=0.1, value=0.5, label="λ accent", show_value=True)
    lambda_similarity = mo.ui.slider(start=0.0, stop=12.0, step=0.5, value=6.0, label="λ similarity", show_value=True)
    variation_budget = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.3, label="Variation budget", show_value=True
    )
    density_amplitude = mo.ui.slider(
        start=0.0, stop=4.0, step=0.1, value=1.0, label="Density amplitude", show_value=True
    )
    density_basis_count = mo.ui.number(start=1, stop=8, step=1, value=2, label="Density oscillations")
    bpm = mo.ui.slider(start=30, stop=240, step=1, value=80, label="BPM", show_value=True)
    notation_bars = mo.ui.slider(start=1, stop=32, step=1, value=8, label="Notation bars", show_value=True)

    def _capture_request(_):
        set_render_request(
            FormRenderRequest(
                scale_root=int(scale_root.value),
                scale_type=scale_type.value,
                time_numerator=int(time_numerator.value),
                time_denominator=int(time_denominator.value),
                bar_count=int(bar_count.value),
                seed=int(seed.value),
                harmonic_slot_denominator=int(harmonic_slot_denominator.value),
                prior_source=prior_source.value,
                commonness_bias=float(commonness_bias.value),
                lambda_curve=float(lambda_curve.value),
                lambda_harmonic=float(lambda_harmonic.value),
                lambda_accent=float(lambda_accent.value),
                lambda_similarity=float(lambda_similarity.value),
                variation_budget=float(variation_budget.value),
                density_amplitude=float(density_amplitude.value),
                density_basis_count=int(density_basis_count.value),
            )
        )

    render_button = mo.ui.run_button(label="Render", on_change=_capture_request)
    mo.vstack(
        [
            mo.md("## Controls"),
            mo.md(
                "**λ harmonic** lands chord tones on strong beats, **λ curve** follows the register arc, "
                "**λ similarity** drives motif reuse across restatements (0 = independent draws / corpus marginal). "
                "**Variation budget** sets how much restatements transform."
            ),
            mo.hstack(
                [scale_root, scale_type, time_numerator, time_denominator, bar_count, seed],
                gap=2,
                wrap=True,
            ),
            mo.hstack([harmonic_slot_denominator, prior_source, bpm, notation_bars], gap=2, wrap=True),
            mo.md("### Figure tilt"),
            mo.hstack(
                [commonness_bias, lambda_curve, lambda_harmonic, lambda_accent, lambda_similarity, variation_budget],
                gap=2,
                wrap=True,
            ),
            mo.md("### Rhythm (density drift of the phrasing tempo)"),
            mo.hstack([density_amplitude, density_basis_count], gap=2, wrap=True),
            render_button,
        ],
        gap=2,
    )
    return bpm, notation_bars, render_button


@app.cell
def _(form_fitting, mo, render_form_segment, render_request, synthetic_inputs):
    request = render_request()
    if request is None or synthetic_inputs is None:
        output = None
    else:
        with mo.status.spinner(title="Rendering form-driven exercise..."):
            output = render_form_segment(synthetic_inputs, request, form_fitting=form_fitting)

    return (output,)


@app.cell
def _(hand_controls, mo):
    output_hand_controls = hand_controls(mo)
    return (output_hand_controls,)


@app.cell
def _(mo, output):
    if output is None or output.form is None:
        form_output = mo.md("")
    else:
        phrase_lines = "\n".join(
            f"- phrase {index}: bars [{phrase.start_bar}, {phrase.start_bar + phrase.bar_span}) — "
            f"closing `{'>'.join(function.value for function in phrase.closing.functions)}`"
            for index, phrase in enumerate(output.form.phrases)
        )
        segment_lines = "\n".join(
            f"- segment {index}: bars [{segment.start_bar}, {segment.start_bar + segment.bar_span}) — "
            f"class {segment.class_label}, {segment.variation.value}"
            for index, segment in enumerate(output.form.segments)
        )
        form_output = mo.md(f"## Sampled FormTree\n**Phrases**\n{phrase_lines}\n\n**Segments**\n{segment_lines}")

    form_output
    return


@app.cell
def _(bpm, mo, notation_bars, output, score_data_html, segment_to_score_data):
    if output is None or output.segment is None:
        notation_output = mo.md("")
    elif output.decode_error is not None:
        notation_output = mo.callout(f"Notation skipped because decoding failed: {output.decode_error}", kind="warn")
    else:
        score_data = segment_to_score_data(
            output.segment,
            duration_vocabulary=output.duration_vocabulary,
            tempo=bpm.value,
            measures_per_row=4,
            max_bars=notation_bars.value,
            layout="grand_staff",
        )
        iframe_height = f"{max(260, len(score_data.rows) * 220 + 24)}px"
        notation_output = mo.iframe(score_data_html(score_data), height=iframe_height)

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
    scale_pitch_class_set,
    segment_piano_roll_view_data,
):
    if output is None or output.segment is None:
        piano_roll_output = mo.md("")
    elif output.decode_error is not None:
        piano_roll_output = mo.callout(
            f"Piano roll skipped because decoding failed: {output.decode_error}", kind="warn"
        )
    else:
        view_data = segment_piano_roll_view_data(
            output.segment,
            duration_vocabulary=output.duration_vocabulary,
            pitch_spelling=PitchSpelling.SHARPS,
            bpm=bpm.value,
            title="Form-driven render piano roll",
        )
        piano_roll_output = piano_roll_player_panel(
            view_data,
            mo=mo,
            alt=alt,
            bpm=bpm.value,
            controls=output_hand_controls,
            scale_pitch_classes=scale_pitch_class_set(output.scale_root, output.scale_type),
            chord_highlights=output.chord_highlights,
            show_chord_labels=True,
        )

    piano_roll_output
    return


@app.cell
def _(mo, output):
    status_output = mo.md("") if output is None else mo.callout(output.status_message, kind=output.status_kind)
    status_output
    return


if __name__ == "__main__":
    app.run()
