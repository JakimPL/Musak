import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide", app_title="Model Output Explorer")


@app.cell
def _():
    from fractions import Fraction
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import torch

    from musak_model.generation.constraints import GenerationConstraints
    from musak_model.paths import DEFAULT_CHECKPOINT_DIR, DEFAULT_PROCESSED_ROOT, TOKENIZATION_CONFIG_PATH
    from musak_model.tokens.schema import ScaleType
    from notebooks.utils import (
        PitchSpelling,
        SamplingOptions,
        empty_prompt,
        hand_controls,
        load_encoded_shard,
        load_trained_model,
        piano_roll_player_panel,
        prompt_from_encoded_sample,
        prompt_from_text,
        sample_autoregressive,
        sampling_result_to_segment,
        score_data_html,
        segment_decode_error,
        segment_event_count,
        segment_piano_roll_view_data,
        segment_to_score_data,
        selected_file,
        token_rows,
        trace_rows,
    )

    alt.data_transformers.disable_max_rows()
    return (
        DEFAULT_CHECKPOINT_DIR,
        DEFAULT_PROCESSED_ROOT,
        Fraction,
        GenerationConstraints,
        Path,
        PitchSpelling,
        SamplingOptions,
        ScaleType,
        TOKENIZATION_CONFIG_PATH,
        alt,
        empty_prompt,
        hand_controls,
        load_encoded_shard,
        load_trained_model,
        mo,
        piano_roll_player_panel,
        prompt_from_encoded_sample,
        prompt_from_text,
        sample_autoregressive,
        sampling_result_to_segment,
        score_data_html,
        segment_decode_error,
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
    DEFAULT_PROCESSED_ROOT,
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
    encoded_browser = mo.ui.file_browser(
        initial_path=DEFAULT_PROCESSED_ROOT,
        filetypes=[".jsonl"],
        selection_mode="file",
        multiple=False,
        label="Encoded shard",
    )
    tokenization_path = mo.ui.text(value=str(TOKENIZATION_CONFIG_PATH), label="Tokenization config")
    device = mo.ui.dropdown(options=["cpu", "cuda"], value="cpu", label="Device")
    seed = mo.ui.number(start=0, stop=2**31 - 1, step=1, value=1234, label="Seed")
    setup_output = mo.vstack(
        [
            mo.md("## Setup"),
            mo.hstack([checkpoint_browser, encoded_browser], gap=2),
            mo.hstack([tokenization_path, device, seed], gap=2),
        ],
        gap=2,
    )
    setup_output
    return checkpoint_browser, device, encoded_browser, seed, tokenization_path


@app.cell
def _(
    Path,
    checkpoint_browser,
    device,
    load_trained_model,
    mo,
    selected_file,
    tokenization_path,
):
    checkpoint_selection = selected_file(
        checkpoint_browser,
        supported_suffixes=frozenset({".pt"}),
        description="checkpoint",
    )
    checkpoint_path = checkpoint_selection.path
    if checkpoint_path is None:
        loaded_model = None
        setup_status = mo.callout(checkpoint_selection.message or "Select a checkpoint to load a model.", kind="warn")
    else:
        with mo.status.spinner(title="Loading model checkpoint..."):
            loaded_model = load_trained_model(
                checkpoint_path,
                device=device.value,
                tokenization_config_path=Path(tokenization_path.value),
            )
        setup_status = mo.callout(
            (
                f"Loaded `{checkpoint_path.name}` | "
                f"vocab={loaded_model.token_vocabulary.vocabulary_size} | "
                f"max_seq={loaded_model.config.transformer.max_sequence_length} | "
                f"epoch={loaded_model.checkpoint_epoch}"
            ),
            kind="success",
        )

    setup_status
    return (loaded_model,)


@app.cell
def _(encoded_browser, load_encoded_shard, mo, selected_file):
    encoded_selection = selected_file(
        encoded_browser,
        supported_suffixes=frozenset({".jsonl"}),
        description="encoded shard",
    )
    encoded_path = encoded_selection.path
    if encoded_path is None:
        encoded_shard = None
        encoded_status = mo.callout(encoded_selection.message, kind="warn") if encoded_selection.message else mo.md("")
    else:
        with mo.status.spinner(title="Loading encoded shard..."):
            encoded_shard = load_encoded_shard(encoded_path)
        encoded_status = mo.callout(f"Loaded {len(encoded_shard.samples)} encoded sample(s).", kind="success")

    encoded_status
    return (encoded_shard,)


@app.cell
def _(ScaleType, encoded_shard, mo):
    prompt_source = mo.ui.radio(
        options={"Empty": "empty", "Encoded sample": "encoded", "Pasted token text": "text"},
        value="Empty",
        inline=True,
        label="Prompt source",
    )
    sample_slider = (
        mo.ui.slider(start=0, stop=len(encoded_shard.samples) - 1, step=1, value=0, label="Sample")
        if encoded_shard is not None and encoded_shard.samples
        else None
    )
    pasted_tokens = mo.ui.text_area(value="", label="Token text")
    max_new_tokens = mo.ui.slider(start=1, stop=512, step=1, value=128, label="Max new tokens")
    temperature = mo.ui.slider(start=0.1, stop=2.0, step=0.05, value=1.0, label="Temperature")
    top_k = mo.ui.number(start=0, stop=512, step=1, value=0, label="Top-k (0 disables)")
    top_p = mo.ui.slider(start=0.05, stop=1.0, step=0.05, value=1.0, label="Top-p")
    greedy = mo.ui.checkbox(value=False, label="Greedy")
    key_root = mo.ui.slider(start=0, stop=11, step=1, value=0, label="Key root")
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
    max_notes_per_onset = mo.ui.number(start=0, stop=8, step=1, value=0, label="Max notes per onset (0 disables)")
    max_gap = mo.ui.number(start=0, stop=36, step=1, value=0, label="Max melodic gap (0 disables)")
    max_span = mo.ui.number(start=0, stop=21, step=1, value=0, label="Static hand span (0 disables)")
    generate_button = mo.ui.run_button(label="Generate")
    prompt_output = mo.vstack(
        [
            mo.md("## Prompt and Controls"),
            prompt_source,
            sample_slider if sample_slider is not None else mo.md(""),
            pasted_tokens,
            mo.hstack([max_new_tokens, temperature, top_k, top_p, greedy], gap=2),
            mo.hstack([key_root, scale_type, time_numerator, time_denominator, target_bars, bpm, notation_bars], gap=2),
            mo.hstack([use_constraints, minimum_duration, allow_dotted, max_notes_per_onset, max_gap, max_span], gap=2),
            generate_button,
        ],
        gap=2,
    )
    prompt_output
    return (
        allow_dotted,
        bpm,
        generate_button,
        greedy,
        key_root,
        max_gap,
        max_new_tokens,
        max_notes_per_onset,
        max_span,
        minimum_duration,
        notation_bars,
        pasted_tokens,
        prompt_source,
        sample_slider,
        scale_type,
        target_bars,
        temperature,
        time_denominator,
        time_numerator,
        top_k,
        top_p,
        use_constraints,
    )


@app.cell
def _(
    Fraction,
    GenerationConstraints,
    SamplingOptions,
    ScaleType,
    allow_dotted,
    empty_prompt,
    encoded_shard,
    generate_button,
    greedy,
    key_root,
    loaded_model,
    max_gap,
    max_new_tokens,
    max_notes_per_onset,
    max_span,
    minimum_duration,
    mo,
    pasted_tokens,
    prompt_from_encoded_sample,
    prompt_from_text,
    prompt_source,
    sample_autoregressive,
    sample_slider,
    sampling_result_to_segment,
    scale_type,
    seed,
    segment_decode_error,
    target_bars,
    temperature,
    time_denominator,
    time_numerator,
    top_k,
    top_p,
    use_constraints,
):
    if loaded_model is None:
        prompt = None
        sampling_result = None
        decoded_segment = None
        decode_error = None
        generation_status = mo.md("")
    elif not generate_button.value:
        prompt = None
        sampling_result = None
        decoded_segment = None
        decode_error = None
        generation_status = mo.callout("Adjust controls, then click Generate.", kind="warn")
    else:
        if prompt_source.value == "encoded" and encoded_shard is not None and sample_slider is not None:
            prompt = prompt_from_encoded_sample(
                encoded_shard.samples[sample_slider.value],
                token_vocabulary=loaded_model.token_vocabulary,
                duration_vocabulary=loaded_model.duration_vocabulary,
            )
        elif prompt_source.value == "text":
            prompt = prompt_from_text(
                pasted_tokens.value,
                token_vocabulary=loaded_model.token_vocabulary,
                duration_vocabulary=loaded_model.duration_vocabulary,
            )
        else:
            prompt = empty_prompt(
                token_vocabulary=loaded_model.token_vocabulary,
                duration_vocabulary=loaded_model.duration_vocabulary,
            )

        selected_scale_type = ScaleType(scale_type.value)
        hard_constraints = None
        if use_constraints.value:
            hard_constraints = GenerationConstraints(
                time_numerator=int(time_numerator.value),
                time_denominator=int(time_denominator.value),
                bar_count=int(target_bars.value),
                minimum_duration=Fraction(minimum_duration.value) if minimum_duration.value != "None" else None,
                allow_dotted_durations=allow_dotted.value,
                max_notes_per_onset_per_hand=int(max_notes_per_onset.value) or None,
                maximum_pitch_gap_semitones=int(max_gap.value) or None,
                maximum_static_hand_span_degrees=int(max_span.value) or None,
                key_root=int(key_root.value),
                scale_type=selected_scale_type,
            )

        options = SamplingOptions(
            max_new_tokens=int(max_new_tokens.value),
            temperature=float(temperature.value),
            top_k=int(top_k.value) or None,
            top_p=float(top_p.value) if float(top_p.value) < 1.0 else None,
            greedy=greedy.value,
            seed=int(seed.value),
            constraints=hard_constraints,
            scale_type=selected_scale_type,
            time_signature=(int(time_numerator.value), int(time_denominator.value)),
        )
        with mo.status.progress_bar(
            total=int(max_new_tokens.value),
            title="Sampling model output...",
            completion_title="Sampling complete",
        ) as progress:

            def _update_sampling_progress(step, token, stop_reason):
                progress.update(
                    title="Sampling model output...",
                    subtitle=f"step {step}/{int(max_new_tokens.value)} | {token.kind} | {stop_reason}",
                )

            sampling_result = sample_autoregressive(
                loaded_model.model,
                prompt,
                options=options,
                token_vocabulary=loaded_model.token_vocabulary,
                duration_vocabulary=loaded_model.duration_vocabulary,
                model_config=loaded_model.config,
                device=loaded_model.device,
                progress_callback=_update_sampling_progress,
            )
        decoded_segment = sampling_result_to_segment(
            sampling_result,
            key_root=int(key_root.value),
            scale_type=selected_scale_type,
            time_numerator=int(time_numerator.value),
            time_denominator=int(time_denominator.value),
        )
        decode_error = segment_decode_error(decoded_segment, duration_vocabulary=loaded_model.duration_vocabulary)
        generation_status = mo.callout(
            (
                f"Stop reason: `{sampling_result.stop_reason}` | "
                f"new tokens: {sampling_result.generated_token_count} | "
                f"EndToken: {sampling_result.reached_end} | "
                f"constraint error: {sampling_result.constraint_error or '-'} | "
                f"decode error: {decode_error or '-'}"
            ),
            kind="success" if sampling_result.constraint_error is None and decode_error is None else "warn",
        )

    generation_status
    return decode_error, decoded_segment, sampling_result


@app.cell
def _(hand_controls, mo):
    output_hand_controls = hand_controls(mo)
    return (output_hand_controls,)


@app.cell
def _(
    bpm,
    decode_error,
    decoded_segment,
    loaded_model,
    mo,
    notation_bars,
    score_data_html,
    segment_to_score_data,
):
    if decoded_segment is None or loaded_model is None:
        notation_output = mo.md("")
    elif decode_error is not None:
        notation_output = mo.callout(f"Notation skipped because decoding failed: {decode_error}", kind="warn")
    else:
        try:
            score_data = segment_to_score_data(
                decoded_segment,
                duration_vocabulary=loaded_model.duration_vocabulary,
                tempo=bpm.value,
                measures_per_row=4,
                max_bars=notation_bars.value,
            )
            bar_note = (
                mo.callout(
                    f"Showing first {notation_bars.value} of {decoded_segment.bar_count} display bar(s) in notation.",
                    kind="warn",
                )
                if decoded_segment.bar_count > notation_bars.value
                else mo.md("")
            )
            notation_output = mo.vstack([bar_note, mo.Html(score_data_html(score_data))], gap=1)
        except ValueError as exception:
            notation_output = mo.callout(f"Notation rendering unavailable: {exception}", kind="warn")

    notation_output
    return


@app.cell
def _(
    PitchSpelling,
    alt,
    bpm,
    decode_error,
    decoded_segment,
    loaded_model,
    mo,
    output_hand_controls,
    piano_roll_player_panel,
    segment_piano_roll_view_data,
):
    if decoded_segment is None or loaded_model is None:
        piano_roll_output = mo.md("")
    elif decode_error is not None:
        piano_roll_output = mo.callout(f"Piano roll skipped because decoding failed: {decode_error}", kind="warn")
    else:
        view_data = segment_piano_roll_view_data(
            decoded_segment,
            duration_vocabulary=loaded_model.duration_vocabulary,
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
def _(
    decode_error,
    decoded_segment,
    loaded_model,
    mo,
    sampling_result,
    segment_event_count,
    token_rows,
    trace_rows,
):
    if decoded_segment is None or loaded_model is None or sampling_result is None:
        debug_output = mo.md("")
    else:
        token_table = mo.ui.table(
            token_rows(decoded_segment.tokens, duration_vocabulary=loaded_model.duration_vocabulary),
            selection=None,
            label="Generated tokens",
        )
        trace_table = mo.ui.table(trace_rows(sampling_result), selection=None, label="Sample trace")
        raw_text = " ".join(row["token"] for row in trace_rows(sampling_result))
        summary_rows = [
            {"metric": "display bars", "value": decoded_segment.bar_count},
            {"metric": "tokens total", "value": len(decoded_segment.tokens)},
            {"metric": "generated tokens", "value": sampling_result.generated_token_count},
            {
                "metric": "decoded note events",
                "value": (
                    segment_event_count(decoded_segment, duration_vocabulary=loaded_model.duration_vocabulary)
                    if decode_error is None
                    else "-"
                ),
            },
            {"metric": "stop reason", "value": sampling_result.stop_reason},
            {"metric": "decode error", "value": decode_error or "-"},
        ]
        debug_output = mo.vstack(
            [
                mo.md("## Diagnostics"),
                mo.ui.table(summary_rows, selection=None, label="Summary"),
                token_table,
                mo.accordion(
                    {
                        "Probability trace": trace_table,
                        "Raw sampled token text": mo.md(f"```text\n{raw_text}\n```"),
                        "Decoded metadata": mo.md(f"```text\n{decoded_segment.metadata}\n```"),
                    }
                ),
            ],
            gap=2,
        )

    debug_output
    return


if __name__ == "__main__":
    app.run()
