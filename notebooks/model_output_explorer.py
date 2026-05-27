from __future__ import annotations

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide", app_title="Model Output Explorer")


@app.cell
def _():
    from fractions import Fraction

    import altair as alt
    import marimo as mo
    import torch

    from musak_model.conditioning.structural.schema import StructuralControlFeatures
    from musak_model.decoder.notation import segment_to_score_data
    from musak_model.generation.constraints import GenerationConstraints
    from musak_model.paths import DEFAULT_CHECKPOINT_DIR, DEFAULT_TRAINING_FIGURE_DIR, TOKENIZATION_CONFIG_PATH
    from musak_model.tokens.schema import ScaleType
    from musak_shared.notation.html import score_data_html
    from notebooks.utils import (
        GeneratedOutput,
        GenerationRequest,
        LoadedModel,
        PitchSpelling,
        SamplingOptions,
        empty_prompt,
        figure_pattern_metric_rows,
        figure_reference_alignment_metric_rows,
        figure_reference_count_groups,
        generation_summary_metric_rows,
        hand_controls,
        load_figure_reference_counts,
        load_trained_model,
        piano_roll_player_panel,
        prompt_from_text,
        rhythm_grid_metric_rows,
        sample_autoregressive,
        sampling_result_to_segment,
        segment_decode_error,
        segment_diagnostic_rows,
        segment_event_count,
        segment_piano_roll_view_data,
        selected_file,
        token_rows,
        trace_rows,
    )

    alt.data_transformers.disable_max_rows()
    return (
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_TRAINING_FIGURE_DIR,
        Fraction,
        GeneratedOutput,
        GenerationRequest,
        GenerationConstraints,
        PitchSpelling,
        SamplingOptions,
        ScaleType,
        StructuralControlFeatures,
        TOKENIZATION_CONFIG_PATH,
        alt,
        empty_prompt,
        figure_pattern_metric_rows,
        figure_reference_alignment_metric_rows,
        figure_reference_count_groups,
        generation_summary_metric_rows,
        hand_controls,
        load_figure_reference_counts,
        load_trained_model,
        mo,
        piano_roll_player_panel,
        prompt_from_text,
        rhythm_grid_metric_rows,
        sample_autoregressive,
        sampling_result_to_segment,
        score_data_html,
        segment_decode_error,
        segment_diagnostic_rows,
        segment_event_count,
        segment_piano_roll_view_data,
        segment_to_score_data,
        selected_file,
        token_rows,
        trace_rows,
    )


@app.cell
def _(mo):
    mo.md("""
    # Model Output Explorer
    """)
    return


@app.cell
def _(
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_TRAINING_FIGURE_DIR,
    TOKENIZATION_CONFIG_PATH,
    mo,
):
    checkpoint_browser = mo.ui.file_browser(
        initial_path=DEFAULT_CHECKPOINT_DIR if DEFAULT_CHECKPOINT_DIR.exists() else ".",
        filetypes=[".pt"],
        selection_mode="file",
        multiple=False,
        label="Checkpoint",
    )
    tokenization_browser = mo.ui.file_browser(
        initial_path=TOKENIZATION_CONFIG_PATH.parent,
        filetypes=[".yml", ".yaml"],
        selection_mode="file",
        multiple=False,
        label="Tokenization config",
    )
    reference_counts_browser = mo.ui.file_browser(
        initial_path=DEFAULT_TRAINING_FIGURE_DIR if DEFAULT_TRAINING_FIGURE_DIR.exists() else ".",
        filetypes=[".csv"],
        selection_mode="file",
        multiple=False,
        label="Reference figure counts CSV",
    )
    device = mo.ui.dropdown(options=["cpu", "cuda"], value="cpu", label="Device")
    setup_output = mo.vstack(
        [
            mo.md("## Setup"),
            mo.hstack([checkpoint_browser, tokenization_browser], gap=2, align="end", widths="equal"),
            reference_counts_browser,
            device,
        ],
        gap=2,
    )
    setup_output
    return checkpoint_browser, device, reference_counts_browser, tokenization_browser


@app.cell
def _(
    TOKENIZATION_CONFIG_PATH,
    checkpoint_browser,
    device,
    load_trained_model,
    mo,
    selected_file,
    tokenization_browser,
):
    checkpoint_selection = selected_file(
        checkpoint_browser,
        supported_suffixes=frozenset({".pt"}),
        description="checkpoint",
    )
    tokenization_selection = selected_file(
        tokenization_browser,
        supported_suffixes=frozenset({".yaml", ".yml"}),
        description="tokenization config",
    )
    if tokenization_selection.path is None and tokenization_browser.value:
        tokenization_config_path = None
        tokenization_status = tokenization_selection.message
    else:
        tokenization_config_path = tokenization_selection.path or TOKENIZATION_CONFIG_PATH
        tokenization_status = (
            f"tokenization={tokenization_config_path.name}"
            if tokenization_selection.path is not None
            else f"tokenization={tokenization_config_path.name} (default)"
        )

    checkpoint_path = checkpoint_selection.path
    if tokenization_config_path is None:
        loaded_model = None
        setup_status = mo.callout(tokenization_status, kind="warn")
    elif checkpoint_path is None:
        loaded_model = None
        setup_status = mo.callout(checkpoint_selection.message or "Select a checkpoint to load a model.", kind="warn")
    else:
        try:
            with mo.status.spinner(title="Loading model checkpoint..."):
                loaded_model = load_trained_model(
                    checkpoint_path,
                    device=device.value,
                    tokenization_config_path=tokenization_config_path,
                )
        except RuntimeError as exception:
            if "Error(s) in loading state_dict" not in str(exception):
                raise

            loaded_model = None
            setup_status = mo.callout(
                mo.md(f"""
                    Checkpoint is incompatible with the current model configuration.

                    ```text
                    {exception}
                    ```
                    """),
                kind="danger",
            )
        else:
            setup_status = mo.callout(
                (
                    f"Loaded `{checkpoint_path.name}` | "
                    f"{tokenization_status} | "
                    f"vocab={loaded_model.token_vocabulary.vocabulary_size} | "
                    f"max_seq={loaded_model.config.transformer.max_sequence_length} | "
                    f"epoch={loaded_model.checkpoint_epoch}"
                ),
                kind="success",
            )

    setup_status
    return (loaded_model,)


@app.cell
def _(mo):
    generation_request, set_generation_request = mo.state(None)
    return generation_request, set_generation_request


@app.cell
def _(GenerationRequest, ScaleType, loaded_model, mo, set_generation_request):
    mo.stop(loaded_model is None, mo.md(""))

    prompt_example = "R 1(1:4) 3(1:4) L r(1:2) |"
    pasted_tokens = mo.ui.text_area(
        value="",
        placeholder=prompt_example,
        label="Token text",
        rows=4,
        full_width=True,
    )
    max_sequence_length = loaded_model.config.transformer.max_sequence_length
    max_new_tokens = mo.ui.slider(
        start=1,
        stop=max_sequence_length,
        step=1,
        value=min(128, max_sequence_length),
        label="Max new tokens",
        show_value=True,
    )
    temperature = mo.ui.slider(start=0.1, stop=2.0, step=0.05, value=1.0, label="Temperature")
    top_k = mo.ui.number(start=0, stop=512, step=1, value=0, label="Top-k (0 disables)")
    top_p = mo.ui.slider(start=0.05, stop=1.0, step=0.05, value=1.0, label="Top-p")
    greedy = mo.ui.checkbox(value=False, label="Greedy")
    seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=1234, label="Seed")
    scale_root = mo.ui.slider(start=0, stop=11, step=1, value=0, label="Scale root")
    scale_type = mo.ui.dropdown(
        options=[scale.value for scale in ScaleType], value=ScaleType.MAJOR.value, label="Scale"
    )
    time_numerator = mo.ui.number(start=1, stop=16, step=1, value=4, label="Time numerator")
    time_denominator = mo.ui.dropdown(options=["1", "2", "4", "8", "16"], value="4", label="Time denominator")
    target_bars = mo.ui.number(start=1, stop=32, step=1, value=4, label="Target bars")
    bpm = mo.ui.slider(start=30, stop=240, step=1, value=80, label="BPM")
    notation_bars = mo.ui.slider(start=1, stop=32, step=1, value=8, label="Notation bars")
    use_constraints = mo.ui.checkbox(value=True, label="Hard constraints")
    minimum_duration = mo.ui.dropdown(
        options=["None", "1/16", "1/8", "1/4", "1/2"],
        value="None",
        label="Shortest duration",
    )
    allow_dotted = mo.ui.checkbox(value=True, label="Allow dotted notes")
    max_notes_per_hand = mo.ui.number(start=0, stop=5, step=1, value=5, label="Max notes per hand (0 disables)")
    max_onset_span = mo.ui.number(start=0, stop=12, step=1, value=12, label="Max onset span (semitones)")
    max_gap = mo.ui.number(start=0, stop=36, step=1, value=0, label="Max melodic gap (0 disables)")
    max_span = mo.ui.number(start=0, stop=21, step=1, value=0, label="Static hand span (0 disables)")

    def _capture_generation_request(_):
        set_generation_request(
            GenerationRequest(
                loaded_model=loaded_model,
                prompt_text=pasted_tokens.value,
                max_new_tokens=int(max_new_tokens.value),
                temperature=float(temperature.value),
                top_k=int(top_k.value) or None,
                top_p=float(top_p.value) if float(top_p.value) < 1.0 else None,
                greedy=greedy.value,
                seed=int(seed.value),
                scale_root=int(scale_root.value),
                scale_type=scale_type.value,
                time_numerator=int(time_numerator.value),
                time_denominator=int(time_denominator.value),
                target_bars=int(target_bars.value),
                use_constraints=use_constraints.value,
                minimum_duration=minimum_duration.value,
                allow_dotted=allow_dotted.value,
                max_notes_per_hand=int(max_notes_per_hand.value) or None,
                max_onset_span=int(max_onset_span.value) or None,
                max_gap=int(max_gap.value) or None,
                max_span=int(max_span.value) or None,
            )
        )

    generate_button = mo.ui.run_button(label="Generate", on_change=_capture_generation_request)
    prompt_controls = mo.vstack([mo.md("### Prompt"), pasted_tokens], gap=1)
    sampling_controls = mo.vstack(
        [
            mo.md("### Sampling"),
            mo.hstack([max_new_tokens, temperature, top_k, top_p, greedy, seed], gap=2, align="end", wrap=True),
        ],
        gap=1,
    )
    musical_controls = mo.vstack(
        [
            mo.md("### Musical Context"),
            mo.hstack(
                [scale_root, scale_type, time_numerator, time_denominator, target_bars],
                gap=2,
                align="end",
                wrap=True,
            ),
        ],
        gap=1,
    )
    playback_controls = mo.vstack(
        [
            mo.md("### Playback and Display"),
            mo.hstack([bpm, notation_bars], gap=2, align="end"),
        ],
        gap=1,
    )
    constraint_controls = mo.vstack(
        [
            mo.md("### Hard Constraints"),
            mo.hstack([use_constraints, minimum_duration, allow_dotted], gap=2, align="end", wrap=True),
            mo.hstack([max_notes_per_hand, max_onset_span, max_gap, max_span], gap=2, align="end", wrap=True),
        ],
        gap=1,
    )
    prompt_output = mo.vstack(
        [
            mo.md("## Prompt and Controls"),
            prompt_controls,
            sampling_controls,
            musical_controls,
            playback_controls,
            constraint_controls,
            generate_button,
        ],
        gap=2,
    )
    prompt_output
    return (
        bpm,
        generate_button,
        notation_bars,
    )


@app.cell
def _(
    Fraction,
    GeneratedOutput,
    GenerationConstraints,
    SamplingOptions,
    ScaleType,
    StructuralControlFeatures,
    empty_prompt,
    generation_request,
    mo,
    prompt_from_text,
    sample_autoregressive,
    sampling_result_to_segment,
    segment_decode_error,
):
    request = generation_request()
    if request is None:
        output = None
    else:
        request_model = request.loaded_model
        if request.prompt_text.strip():
            prompt = prompt_from_text(
                request.prompt_text,
                token_vocabulary=request_model.token_vocabulary,
                duration_vocabulary=request_model.duration_vocabulary,
            )
        else:
            prompt = empty_prompt(
                token_vocabulary=request_model.token_vocabulary,
                duration_vocabulary=request_model.duration_vocabulary,
            )

        selected_scale_type = ScaleType(request.scale_type)
        hard_constraints = None
        if request.use_constraints:
            hard_constraints = GenerationConstraints(
                time_numerator=request.time_numerator,
                time_denominator=request.time_denominator,
                bar_count=request.target_bars,
                minimum_duration=Fraction(request.minimum_duration) if request.minimum_duration != "None" else None,
                allow_dotted_durations=request.allow_dotted,
                max_notes_per_hand=request.max_notes_per_hand,
                maximum_onset_span_semitones=request.max_onset_span,
                maximum_pitch_gap_semitones=request.max_gap,
                maximum_static_hand_span_degrees=request.max_span,
                scale_root=request.scale_root,
                scale_type=selected_scale_type,
            )

        options = SamplingOptions(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            greedy=request.greedy,
            seed=request.seed,
            constraints=hard_constraints,
            scale_type=selected_scale_type,
            time_signature=(request.time_numerator, request.time_denominator),
            structural_features=StructuralControlFeatures(
                shortest_note_duration=(
                    Fraction(request.minimum_duration) if request.minimum_duration != "None" else None
                ),
                has_dotted_notes=None if request.allow_dotted else False,
                max_notes_per_onset=None,
                max_notes_per_hand=request.max_notes_per_hand,
                max_onset_span_semitones=request.max_onset_span,
                max_melodic_gap_semitones=request.max_gap,
                static_hand_span_degrees=request.max_span,
                bar_count=request.target_bars,
            ),
        )
        with mo.status.progress_bar(
            total=request.max_new_tokens,
            title="Sampling model output...",
            remove_on_exit=True,
        ) as progress:

            def _update_sampling_progress(step, token, stop_reason):
                progress.update(
                    title="Sampling model output...",
                    subtitle=f"step {step}/{request.max_new_tokens} | {token.kind} | {stop_reason}",
                )

            sampling_result = sample_autoregressive(
                request_model.model,
                prompt,
                options=options,
                token_vocabulary=request_model.token_vocabulary,
                duration_vocabulary=request_model.duration_vocabulary,
                model_config=request_model.config,
                device=request_model.device,
                progress_callback=_update_sampling_progress,
            )
        decoded_segment = sampling_result_to_segment(
            sampling_result,
            scale_root=request.scale_root,
            scale_type=selected_scale_type,
            time_numerator=request.time_numerator,
            time_denominator=request.time_denominator,
        )
        decode_error = segment_decode_error(decoded_segment, duration_vocabulary=request_model.duration_vocabulary)
        status_message = (
            f"Stop reason: `{sampling_result.stop_reason}` | "
            f"new tokens: {sampling_result.generated_token_count} | "
            f"EndToken: {sampling_result.reached_end} | "
            f"constraint error: {sampling_result.constraint_error or '-'} | "
            f"decode error: {decode_error or '-'}"
        )
        status_kind = "success" if sampling_result.constraint_error is None and decode_error is None else "warn"
        output = GeneratedOutput(
            sampling_result=sampling_result,
            decoded_segment=decoded_segment,
            decode_error=decode_error,
            duration_vocabulary=request_model.duration_vocabulary,
            status_message=status_message,
            status_kind=status_kind,
        )

    return (output,)


@app.cell
def _(hand_controls, mo):
    output_hand_controls = hand_controls(mo)
    return (output_hand_controls,)


@app.cell
def _(
    bpm,
    output,
    mo,
    notation_bars,
    score_data_html,
    segment_to_score_data,
):
    if output is None:
        notation_output = mo.md("")
    elif output.decode_error is not None:
        notation_output = mo.callout(f"Notation skipped because decoding failed: {output.decode_error}", kind="warn")
    else:
        try:
            score_data = segment_to_score_data(
                output.decoded_segment,
                duration_vocabulary=output.duration_vocabulary,
                tempo=bpm.value,
                measures_per_row=4,
                max_bars=notation_bars.value,
            )
            bar_note = (
                mo.callout(
                    (
                        f"Showing first {notation_bars.value} of "
                        f"{output.decoded_segment.bar_count} display bar(s) in notation."
                    ),
                    kind="warn",
                )
                if output.decoded_segment.bar_count > notation_bars.value
                else mo.md("")
            )
            iframe_height = f"{max(220, len(score_data.rows) * 140 + 24)}px"
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
    if output is None:
        piano_roll_output = mo.md("")
    elif output.decode_error is not None:
        piano_roll_output = mo.callout(
            f"Piano roll skipped because decoding failed: {output.decode_error}",
            kind="warn",
        )
    else:
        view_data = segment_piano_roll_view_data(
            output.decoded_segment,
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
def _(mo, output):
    generation_status_output = (
        mo.md("") if output is None else mo.callout(output.status_message, kind=output.status_kind)
    )
    generation_status_output
    return


@app.cell
def _(
    mo,
    output,
    figure_pattern_metric_rows,
    figure_reference_alignment_metric_rows,
    figure_reference_count_groups,
    generation_summary_metric_rows,
    load_figure_reference_counts,
    reference_counts_browser,
    rhythm_grid_metric_rows,
    segment_diagnostic_rows,
    segment_event_count,
    selected_file,
    token_rows,
    trace_rows,
):
    if output is None:
        debug_output = mo.md("")
    else:
        token_table = mo.ui.table(
            token_rows(output.decoded_segment.tokens, duration_vocabulary=output.duration_vocabulary),
            selection=None,
            label="Generated tokens",
        )
        trace_table = mo.ui.table(trace_rows(output.sampling_result), selection=None, label="Sample trace")
        raw_text = " ".join(row["token"] for row in trace_rows(output.sampling_result))
        summary_metric_rows = (
            generation_summary_metric_rows(output.decoded_segment, duration_vocabulary=output.duration_vocabulary)
            if output.decode_error is None
            else []
        )
        figure_metric_rows = (
            figure_pattern_metric_rows(output.decoded_segment, duration_vocabulary=output.duration_vocabulary)
            if output.decode_error is None
            else []
        )
        rhythm_metric_rows = (
            rhythm_grid_metric_rows(output.decoded_segment, duration_vocabulary=output.duration_vocabulary)
            if output.decode_error is None
            else []
        )
        reference_status = mo.md("")
        reference_alignment_rows = []
        if output.decode_error is None and reference_counts_browser.value:
            selection = selected_file(
                reference_counts_browser,
                supported_suffixes=frozenset({".csv"}),
                description="reference figure counts CSV",
            )
            if selection.path is None:
                reference_status = mo.callout(selection.message or "Reference counts CSV is unavailable.", kind="warn")
            else:
                try:
                    reference_groups = figure_reference_count_groups(
                        output.decoded_segment,
                        duration_vocabulary=output.duration_vocabulary,
                    )
                    reference_counts = load_figure_reference_counts(
                        selection.path,
                        scale_type=output.decoded_segment.scale_type,
                        groups=reference_groups,
                    )
                    reference_alignment_rows = figure_reference_alignment_metric_rows(
                        output.decoded_segment,
                        duration_vocabulary=output.duration_vocabulary,
                        reference_counts=reference_counts,
                    )
                except ValueError as exception:
                    reference_status = mo.callout(f"Reference counts CSV could not be loaded: {exception}", kind="warn")
                else:
                    reference_status = mo.callout(f"Comparing against `{selection.path.name}`.", kind="success")
        detailed_diagnostic_rows = (
            segment_diagnostic_rows(output.decoded_segment, duration_vocabulary=output.duration_vocabulary)
            if output.decode_error is None
            else []
        )
        summary_rows = [
            {"metric": "display bars", "value": output.decoded_segment.bar_count},
            {"metric": "tokens total", "value": len(output.decoded_segment.tokens)},
            {"metric": "generated tokens", "value": output.sampling_result.generated_token_count},
            {
                "metric": "decoded note events",
                "value": (
                    segment_event_count(output.decoded_segment, duration_vocabulary=output.duration_vocabulary)
                    if output.decode_error is None
                    else "-"
                ),
            },
            {"metric": "stop reason", "value": output.sampling_result.stop_reason},
            {"metric": "decode error", "value": output.decode_error or "-"},
        ]
        debug_output = mo.vstack(
            [
                mo.md("## Generated Music Summary"),
                mo.ui.table(summary_metric_rows, selection=None, label="Generated music summary"),
                mo.md("## Figure Patterns"),
                mo.ui.table(figure_metric_rows, selection=None, label="Generated figure patterns"),
                mo.md("## Rhythm Grid"),
                mo.ui.table(rhythm_metric_rows, selection=None, label="Generated rhythm grid"),
                mo.md("## Reference Alignment"),
                reference_status,
                mo.ui.table(reference_alignment_rows, selection=None, label="Reference alignment and novelty"),
                mo.md("## Diagnostics"),
                mo.ui.table(summary_rows, selection=None, label="Summary"),
                token_table,
                mo.accordion(
                    {
                        "Detailed musical diagnostics": mo.ui.table(
                            detailed_diagnostic_rows,
                            selection=None,
                            label="Musical diagnostics",
                        ),
                        "Probability trace": trace_table,
                        "Raw sampled token text": mo.md(f"```text\n{raw_text}\n```"),
                        "Decoded metadata": mo.md(f"```text\n{output.decoded_segment.metadata}\n```"),
                    }
                ),
            ],
            gap=2,
        )

    debug_output
    return


if __name__ == "__main__":
    app.run()
