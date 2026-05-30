import marimo

__generated_with = "0.23.6"
app = marimo.App(width="wide", app_title="Dataset Quality")


@app.cell
def _():
    from pathlib import Path
    from types import SimpleNamespace

    import altair as alt
    import marimo as mo
    import pandas as pd

    from musak_model.decoder.notation import segment_to_score_data
    from musak_model.paths import DEFAULT_DATA_DIR, DEFAULT_PROCESSED_ROOT
    from musak_model.processing.fingerprint import file_sha256
    from musak_model.processing.manifest import EncodedManifestField
    from musak_model.tokens.schema import Hand
    from musak_shared.notation.html import score_data_html
    from notebooks.utils import (
        DEFAULT_DATASET_QUALITY_DATABASE_PATH,
        PitchSpelling,
        SegmentRating,
        SegmentReviewDecision,
        SourceFileReview,
        build_encoded_jsonl_index,
        eligible_source_rows,
        encoded_run_directories,
        encoded_sample_to_segment,
        encoded_shard_path_for_manifest_row,
        hand_controls,
        initialize_quality_database,
        load_dataset_statistics,
        load_encoded_manifest_selection,
        load_encoded_sample_from_index,
        load_encoded_shard_context,
        mark_source_file_skipped,
        piano_roll_audio_data,
        piano_roll_audio_panel,
        piano_roll_chart_panel,
        processed_dataset_directories,
        quality_database_summary_rows,
        rating_by_segment_key,
        segment_piano_roll_view_data,
        unrated_source_frame,
        upsert_segment_rating,
    )

    return (
        DEFAULT_DATASET_QUALITY_DATABASE_PATH,
        DEFAULT_DATA_DIR,
        DEFAULT_PROCESSED_ROOT,
        EncodedManifestField,
        Hand,
        Path,
        PitchSpelling,
        SegmentRating,
        SegmentReviewDecision,
        SimpleNamespace,
        SourceFileReview,
        alt,
        build_encoded_jsonl_index,
        eligible_source_rows,
        encoded_sample_to_segment,
        encoded_run_directories,
        encoded_shard_path_for_manifest_row,
        file_sha256,
        hand_controls,
        initialize_quality_database,
        load_dataset_statistics,
        load_encoded_sample_from_index,
        load_encoded_manifest_selection,
        load_encoded_shard_context,
        mark_source_file_skipped,
        mo,
        pd,
        piano_roll_audio_data,
        piano_roll_audio_panel,
        piano_roll_chart_panel,
        processed_dataset_directories,
        quality_database_summary_rows,
        rating_by_segment_key,
        score_data_html,
        segment_piano_roll_view_data,
        segment_to_score_data,
        unrated_source_frame,
        upsert_segment_rating,
    )


@app.cell
def _(mo):
    mo.md("""
    # Dataset Quality

    Review eligible parsed segments from processed PDMX-style datasets.
    """)
    return


@app.cell
def _(
    DEFAULT_PROCESSED_ROOT,
    mo,
    processed_dataset_directories,
):
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
    dataset_name = dataset_dir.name if dataset_dir is not None else ""
    return dataset_dir, dataset_name


@app.cell
def _(DEFAULT_DATASET_QUALITY_DATABASE_PATH, DEFAULT_DATA_DIR, dataset_name, mo):
    default_dataset_root = DEFAULT_DATA_DIR / dataset_name if dataset_name else DEFAULT_DATA_DIR
    dataset_root_text = mo.ui.text(value=default_dataset_root.as_posix(), label="Original dataset root")
    database_path_text = mo.ui.text(
        value=DEFAULT_DATASET_QUALITY_DATABASE_PATH.as_posix(),
        label="Rating database",
    )
    path_controls_output = mo.hstack([dataset_root_text, database_path_text], gap=2, align="end", widths="equal")
    path_controls_output
    return database_path_text, dataset_root_text


@app.cell
def _(Path, database_path_text, dataset_root_text):
    dataset_root = Path(dataset_root_text.value).expanduser()
    database_path = Path(database_path_text.value).expanduser()
    return database_path, dataset_root


@app.cell
def _(dataset_dir, encoded_run_directories, mo):
    if dataset_dir is None:
        encoded_selector = mo.ui.dropdown(options={}, label="Tokenizer run", searchable=True)
        encoded_output = mo.callout("No processed dataset is available.", kind="warn")
    else:
        encoded_directories = encoded_run_directories(dataset_dir)
        encoded_options = {path.name: str(path) for path in encoded_directories}
        encoded_selector = mo.ui.dropdown(
            options=encoded_options,
            value=next(iter(encoded_options), None),
            label="Tokenizer run",
            searchable=True,
        )
        encoded_output = (
            encoded_selector
            if encoded_options
            else mo.callout("No encoded manifest found for this dataset.", kind="warn")
        )
    encoded_output
    return (encoded_selector,)


@app.cell
def _(Path, encoded_selector):
    encoded_directory = Path(encoded_selector.value) if encoded_selector.value is not None else None
    return (encoded_directory,)


@app.cell
def _(database_path, dataset_dir, encoded_directory, initialize_quality_database, load_dataset_statistics, mo, pd):
    stats = None
    load_error = ""
    if dataset_dir is not None and encoded_directory is not None:
        try:
            initialize_quality_database(database_path)
            stats = load_dataset_statistics(dataset_dir, encoded_directory)
        except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as exception:
            load_error = f"{type(exception).__name__}: {exception}"

    if dataset_dir is None:
        load_output = mo.callout("Select a processed dataset.", kind="warn")
    elif encoded_directory is None:
        load_output = mo.callout("Select an encoded run.", kind="warn")
    elif load_error:
        load_output = mo.callout(load_error, kind="danger")
    else:
        load_output = mo.callout(f"Loaded `{dataset_dir}` with ratings at `{database_path}`.", kind="success")
    load_output
    return (stats,)


@app.cell
def _(
    database_path,
    dataset_name,
    mo,
    progress_review_revision,
    quality_database_summary_rows,
    set_progress_review_revision,
    stats,
    unrated_source_frame,
):
    progress_review_revision()
    if stats is None or stats.encoded is None:
        unrated_sources = None
        progress_output = mo.md("")
    else:

        def _refresh_progress(_):
            set_progress_review_revision(progress_review_revision() + 1)

        refresh_progress_button = mo.ui.run_button(label="Refresh progress", on_change=_refresh_progress)
        unrated_sources = unrated_source_frame(
            stats.encoded,
            dataset_name=dataset_name,
            database_path=database_path,
        )
        summary_rows = quality_database_summary_rows(database_path, dataset_name=dataset_name)
        progress_output = mo.vstack(
            [
                mo.ui.table(summary_rows, selection=None, label="Rating database"),
                mo.callout(f"{len(unrated_sources)} source file(s) still have unrated eligible segments.", kind="info"),
                refresh_progress_button,
            ],
            gap=1,
        )
    progress_output
    return (unrated_sources,)


@app.cell
def _(mo):
    selected_source_id, set_selected_source_id = mo.state(None)
    selected_segment_index, set_selected_segment_index = mo.state(0)
    action_message, set_action_message = mo.state("")
    current_source_review_revision, set_current_source_review_revision = mo.state(0)
    progress_review_revision, set_progress_review_revision = mo.state(0)
    completed_source_ids, set_completed_source_ids = mo.state(frozenset())
    return (
        action_message,
        completed_source_ids,
        current_source_review_revision,
        progress_review_revision,
        selected_segment_index,
        selected_source_id,
        set_action_message,
        set_completed_source_ids,
        set_current_source_review_revision,
        set_progress_review_revision,
        set_selected_segment_index,
        set_selected_source_id,
    )


@app.cell
def _(EncodedManifestField, mo, stats):
    if stats is None or stats.encoded is None:
        manual_source_selector = mo.ui.dropdown(options={}, label="Source file", searchable=True)
    else:
        eligible = stats.encoded.loc[stats.encoded[EncodedManifestField.ELIGIBLE_FOR_TRAINING] == True].copy()
        source_options = {
            str(row[EncodedManifestField.SOURCE_PATH]): str(row[EncodedManifestField.SOURCE_ID])
            for _, row in eligible.drop_duplicates(subset=[EncodedManifestField.SOURCE_ID]).iterrows()
        }
        manual_source_selector = mo.ui.dropdown(
            options=source_options,
            value=next(iter(source_options), None),
            label="Source file",
            searchable=True,
        )
    return (manual_source_selector,)


@app.cell
def _(
    completed_source_ids,
    set_completed_source_ids,
    set_selected_segment_index,
    set_selected_source_id,
    unrated_sources,
):
    def draw_next_unrated_source(*, completed_source_id: str | None = None) -> str | None:
        if unrated_sources is None or unrated_sources.empty:
            set_selected_source_id(None)
            return None

        completed_ids = set(completed_source_ids())
        if completed_source_id is not None:
            completed_ids.add(completed_source_id)
            set_completed_source_ids(frozenset(completed_ids))

        candidates = unrated_sources
        if completed_ids:
            candidates = candidates.loc[~candidates["source_id"].astype(str).isin(completed_ids)]
        if candidates.empty:
            set_selected_source_id(None)
            return None

        selected = candidates.sample(n=1).iloc[0]
        set_selected_source_id(str(selected["source_id"]))
        set_selected_segment_index(0)
        return str(selected["source_path"])

    return (draw_next_unrated_source,)


@app.cell
def _(
    action_message,
    draw_next_unrated_source,
    manual_source_selector,
    mo,
    set_action_message,
    set_selected_segment_index,
    set_selected_source_id,
):
    def _draw_random_file(_):
        selected_path = draw_next_unrated_source()
        if selected_path is None:
            set_action_message("No unrated eligible source files remain.")
            return

        set_action_message(f"Selected `{selected_path}`.")

    def _load_manual_file(_):
        set_selected_source_id(manual_source_selector.value)
        set_selected_segment_index(0)
        set_action_message(f"Selected `{manual_source_selector.value}`.")

    random_button = mo.ui.run_button(label="Draw random unrated file", on_change=_draw_random_file)
    load_manual_button = mo.ui.run_button(label="Load selected file", on_change=_load_manual_file)
    message = action_message() if action_message() else ""
    selection_output = mo.vstack(
        [
            mo.hstack([random_button, manual_source_selector, load_manual_button], gap=2, align="end", wrap=True),
            mo.md(message),
        ],
        gap=1,
    )
    selection_output
    return


@app.cell
def _(eligible_source_rows, selected_source_id, stats):
    current_source_id = selected_source_id()
    if stats is None or stats.encoded is None or current_source_id is None:
        selected_rows = None
    else:
        selected_rows = eligible_source_rows(stats.encoded, source_id=current_source_id)
    return current_source_id, selected_rows


@app.cell
def _(
    SourceFileReview,
    current_source_id,
    database_path,
    dataset_name,
    dataset_root,
    file_sha256,
    mark_source_file_skipped,
    mo,
    selected_rows,
    completed_source_ids,
    set_action_message,
    set_completed_source_ids,
):
    if selected_rows is None or selected_rows.empty:
        source_output = mo.md("")
    else:
        first_row = selected_rows.iloc[0]
        selected_source_path = str(first_row["source_path"])
        source_file = dataset_root / selected_source_path
        source_sha256 = file_sha256(source_file) if source_file.exists() else None

        def _skip_file(_):
            mark_source_file_skipped(
                database_path,
                SourceFileReview(
                    dataset_name=dataset_name,
                    dataset_root=dataset_root,
                    source_id=str(current_source_id),
                    source_path=selected_source_path,
                    source_sha256=source_sha256,
                ),
            )
            set_action_message(f"Skipped file `{selected_source_path}`.")
            set_completed_source_ids(frozenset(set(completed_source_ids()) | {str(current_source_id)}))

        skip_button = mo.ui.run_button(label="Skip entire file", on_change=_skip_file)
        source_output = mo.vstack(
            [
                mo.md(f"## `{selected_source_path}`"),
                mo.hstack(
                    [
                        mo.ui.table(
                            [
                                {"Metric": "Source ID", "Value": str(current_source_id)},
                                {"Metric": "Eligible segments", "Value": str(len(selected_rows))},
                                {"Metric": "Source SHA-256", "Value": source_sha256 or "unavailable"},
                            ],
                            selection=None,
                            label="Selected source",
                        ),
                        skip_button,
                    ],
                    gap=2,
                    align="end",
                    wrap=True,
                ),
            ],
            gap=2,
        )
    source_output
    return


@app.cell
def _(current_source_review_revision, database_path, dataset_name, current_source_id, rating_by_segment_key):
    current_source_review_revision()
    existing_ratings = (
        rating_by_segment_key(database_path, dataset_name=dataset_name, source_id=str(current_source_id))
        if current_source_id is not None
        else {}
    )
    return (existing_ratings,)


@app.cell
def _(hand_controls, mo):
    segment_hand_controls = hand_controls(mo)
    return (segment_hand_controls,)


@app.cell
def _(mo):
    render_piano_rolls_checkbox = mo.ui.checkbox(value=False, label="Render piano rolls")
    render_controls_output = render_piano_rolls_checkbox
    render_controls_output
    return (render_piano_rolls_checkbox,)


@app.cell
def _(mo, selected_rows, selected_segment_index):
    if selected_rows is None or selected_rows.empty:
        selected_segment_count = 0
        segment_slider = None
        navigation_output = mo.md("")
    else:
        selected_segment_count = len(selected_rows)
        current_value = min(max(int(selected_segment_index()) + 1, 1), selected_segment_count)
        segment_slider = mo.ui.slider(
            start=1,
            stop=selected_segment_count,
            step=1,
            value=current_value,
            label="Segment",
            show_value=True,
            include_input=True,
        )
        navigation_output = segment_slider
    navigation_output
    return segment_slider, selected_segment_count


@app.cell
def _(segment_slider):
    selected_preview_segment_index = int(segment_slider.value) - 1 if segment_slider is not None else None
    return (selected_preview_segment_index,)


@app.cell
def _(selected_preview_segment_index, selected_rows):
    if selected_rows is None or selected_rows.empty or selected_preview_segment_index is None:
        selected_row_dict = None
    else:
        selected_row = selected_rows.reset_index(drop=True).iloc[selected_preview_segment_index]
        selected_row_dict = {str(key): value for key, value in selected_row.to_dict().items()}
    return (selected_row_dict,)


@app.cell
def _(build_encoded_jsonl_index, load_encoded_sample_from_index, load_encoded_shard_context, mo):
    @mo.cache
    def cached_encoded_shard_context(path):
        return load_encoded_shard_context(path)

    @mo.cache
    def cached_encoded_jsonl_index(path):
        return build_encoded_jsonl_index(path)

    def cached_encoded_sample(path, line_index: int):
        return load_encoded_sample_from_index(cached_encoded_jsonl_index(path), line_index)

    return cached_encoded_sample, cached_encoded_shard_context


@app.cell
def _(
    EncodedManifestField,
    SimpleNamespace,
    cached_encoded_sample,
    cached_encoded_shard_context,
    dataset_dir,
    encoded_directory,
    encoded_sample_to_segment,
    encoded_shard_path_for_manifest_row,
    load_encoded_manifest_selection,
    selected_row_dict,
):
    if selected_row_dict is None or dataset_dir is None:
        selected_segment_selection = None
        selected_segment_error = ""
    else:
        try:
            encoded_line = selected_row_dict.get(str(EncodedManifestField.ENCODED_LINE))
            if encoded_line is None or encoded_line == "" or encoded_line != encoded_line:
                selected_segment_selection = load_encoded_manifest_selection(
                    selected_row_dict,
                    dataset_dir=dataset_dir,
                    encoded_directory=encoded_directory,
                )
            else:
                encoded_shard_path = encoded_shard_path_for_manifest_row(
                    selected_row_dict,
                    dataset_dir=dataset_dir,
                    encoded_directory=encoded_directory,
                )
                shard = cached_encoded_shard_context(encoded_shard_path)
                encoded_line_index = int(encoded_line)
                sample = cached_encoded_sample(encoded_shard_path, encoded_line_index)
                segment = encoded_sample_to_segment(sample, shard=shard)
                selected_segment_selection = SimpleNamespace(
                    segment=segment,
                    duration_vocabulary=shard.duration_vocabulary,
                )
            selected_segment_error = ""
        except (FileNotFoundError, IndexError, TypeError, ValueError) as exception:
            selected_segment_selection = None
            selected_segment_error = f"{type(exception).__name__}: {exception}"

    return selected_segment_error, selected_segment_selection


@app.cell
def _(
    Hand,
    PitchSpelling,
    mo,
    piano_roll_audio_data,
    score_data_html,
    segment_piano_roll_view_data,
    segment_to_score_data,
):
    @mo.cache
    def cached_notation_preview(segment, duration_vocabulary):
        score_data = segment_to_score_data(
            segment,
            duration_vocabulary=duration_vocabulary,
            tempo=60,
            measures_per_row=4,
        )
        return score_data_html(score_data), f"{max(220, len(score_data.rows) * 140 + 24)}px"

    @mo.cache
    def cached_piano_roll_view_data(segment, duration_vocabulary):
        return segment_piano_roll_view_data(
            segment,
            duration_vocabulary=duration_vocabulary,
            pitch_spelling=PitchSpelling.SHARPS,
            bpm=60,
        )

    @mo.cache
    def cached_audio_data(view_data, hand_values: tuple[str, ...]):
        return piano_roll_audio_data(
            view_data,
            bpm=60,
            hands=frozenset(Hand(hand_value) for hand_value in hand_values),
        )

    return cached_audio_data, cached_notation_preview, cached_piano_roll_view_data


@app.cell
def _(
    EncodedManifestField,
    SegmentRating,
    SegmentReviewDecision,
    current_source_id,
    current_source_review_revision,
    database_path,
    dataset_name,
    draw_next_unrated_source,
    existing_ratings,
    mo,
    selected_row_dict,
    selected_segment_error,
    selected_preview_segment_index,
    selected_segment_selection,
    selected_segment_count,
    selected_rows,
    set_action_message,
    set_current_source_review_revision,
    set_selected_segment_index,
    upsert_segment_rating,
):
    if selected_rows is None or selected_rows.empty:
        segment_output = mo.callout("Draw or load a source file to review its eligible segments.", kind="warn")
    elif selected_segment_error:
        segment_output = mo.callout(f"Selected file preview failed: {selected_segment_error}", kind="warn")
    elif selected_segment_selection is None or selected_row_dict is None:
        segment_output = mo.md("")
    elif selected_preview_segment_index is None:
        segment_output = mo.callout("Select a segment to preview.", kind="warn")
    else:
        row_dict = selected_row_dict
        window_start_bar = int(row_dict[str(EncodedManifestField.WINDOW_START_BAR)])
        bar_count = int(row_dict[str(EncodedManifestField.BAR_COUNT)])
        rating_key = (window_start_bar, bar_count)
        existing = existing_ratings.get(rating_key, {})
        existing_rating = str(existing["rating"]) if "rating" in existing else None
        existing_decision = str(existing.get("decision", SegmentReviewDecision.OK.value))
        existing_time_error = bool(existing.get("time_signature_error", 0))
        existing_key_error = bool(existing.get("key_signature_error", 0))

        rating_control = mo.ui.radio(
            options=["1", "2", "3", "4"],
            value=existing_rating,
            inline=True,
            label="Rating",
        )
        decision_control = mo.ui.radio(
            options=[decision.value for decision in SegmentReviewDecision],
            value=existing_decision,
            inline=True,
            label="Decision",
        )
        time_error_control = mo.ui.checkbox(value=existing_time_error, label="Time signature error")
        key_error_control = mo.ui.checkbox(value=existing_key_error, label="Key signature error")

        def _save_segment(
            _,
            *,
            selected_row: dict[str, object] = row_dict,
            selected_rating=rating_control,
            selected_decision=decision_control,
            selected_time_error=time_error_control,
            selected_key_error=key_error_control,
        ):
            if selected_rating.value is None:
                set_action_message("Select a rating before saving this segment.")
                return

            upsert_segment_rating(
                database_path,
                SegmentRating(
                    dataset_name=dataset_name,
                    source_id=str(selected_row[str(EncodedManifestField.SOURCE_ID)]),
                    source_path=str(selected_row[str(EncodedManifestField.SOURCE_PATH)]),
                    window_start_bar=int(selected_row[str(EncodedManifestField.WINDOW_START_BAR)]),
                    bar_count=int(selected_row[str(EncodedManifestField.BAR_COUNT)]),
                    rating=int(selected_rating.value),
                    decision=SegmentReviewDecision(str(selected_decision.value)),
                    time_signature_error=bool(selected_time_error.value),
                    key_signature_error=bool(selected_key_error.value),
                    manifest_segment_id=str(selected_row[str(EncodedManifestField.SEGMENT_ID)]),
                ),
            )
            set_action_message(
                (
                    f"Saved segment {selected_row[str(EncodedManifestField.WINDOW_START_BAR)]}"
                    f"+{selected_row[str(EncodedManifestField.BAR_COUNT)]}."
                )
            )
            set_current_source_review_revision(current_source_review_revision() + 1)
            if selected_preview_segment_index + 1 < selected_segment_count:
                set_selected_segment_index(selected_preview_segment_index + 1)
            else:
                selected_path = draw_next_unrated_source(completed_source_id=str(current_source_id))
                if selected_path is None:
                    set_action_message("Saved final segment. No unrated eligible source files remain.")
                else:
                    set_action_message(f"Saved final segment. Selected `{selected_path}`.")

        save_button = mo.ui.run_button(label="Save and next", on_change=_save_segment)
        status_text = "rated" if rating_key in existing_ratings else "unrated"
        key_signature = row_dict.get(str(EncodedManifestField.DECLARED_KEY_FIFTHS), "")
        key_signature_text = "missing" if key_signature == "" else str(key_signature)
        segment_output = mo.vstack(
            [
                mo.md(
                    (
                        f"### Segment {selected_preview_segment_index + 1}: bars "
                        f"{window_start_bar}-{window_start_bar + bar_count - 1} ({status_text})"
                    )
                ),
                mo.ui.table(
                    [
                        {"Field": "Time signature", "Value": str(row_dict[str(EncodedManifestField.TIME_SIGNATURE)])},
                        {"Field": "Declared key fifths", "Value": key_signature_text},
                        {"Field": "Scale root", "Value": str(row_dict[str(EncodedManifestField.SCALE_ROOT)])},
                        {"Field": "Scale type", "Value": str(row_dict[str(EncodedManifestField.SCALE_TYPE)])},
                        {"Field": "Segment ID", "Value": str(row_dict[str(EncodedManifestField.SEGMENT_ID)])},
                    ],
                    selection=None,
                    label="Review metadata",
                ),
                mo.hstack(
                    [
                        rating_control,
                        decision_control,
                        time_error_control,
                        key_error_control,
                        save_button,
                    ],
                    gap=2,
                    align="end",
                    wrap=True,
                ),
            ],
            gap=1,
        )

    segment_output
    return


@app.cell
def _(cached_notation_preview, mo, selected_segment_error, selected_segment_selection):
    if selected_segment_selection is None or selected_segment_error:
        notation_output = mo.md("")
    else:
        try:
            notation_html, iframe_height = cached_notation_preview(
                selected_segment_selection.segment,
                selected_segment_selection.duration_vocabulary,
            )
            notation_output = mo.iframe(notation_html, height=iframe_height)
        except (FileNotFoundError, IndexError, TypeError, ValueError) as exception:
            notation_output = mo.callout(
                f"Notation preview failed: {type(exception).__name__}: {exception}",
                kind="warn",
            )
    notation_output
    return


@app.cell
def _(
    Hand,
    cached_audio_data,
    cached_piano_roll_view_data,
    mo,
    piano_roll_audio_panel,
    segment_hand_controls,
    selected_segment_error,
    selected_segment_selection,
):
    if selected_segment_selection is None or selected_segment_error:
        audio_output = mo.md("")
    else:
        selected_hand_values = tuple(hand.value for hand in Hand if hand in segment_hand_controls.selected_hands)
        if selected_hand_values:
            audio_output = mo.lazy(
                lambda: piano_roll_audio_panel(
                    cached_piano_roll_view_data(
                        selected_segment_selection.segment,
                        selected_segment_selection.duration_vocabulary,
                    ),
                    mo=mo,
                    bpm=60,
                    controls=segment_hand_controls,
                    audio_data=cached_audio_data(
                        cached_piano_roll_view_data(
                            selected_segment_selection.segment,
                            selected_segment_selection.duration_vocabulary,
                        ),
                        selected_hand_values,
                    ),
                ),
                show_loading_indicator=True,
            )
        else:
            audio_output = mo.callout("Select at least one hand to prepare the player.", kind="warn")
    audio_output
    return


@app.cell
def _(
    alt,
    cached_piano_roll_view_data,
    mo,
    piano_roll_chart_panel,
    render_piano_rolls_checkbox,
    segment_hand_controls,
    selected_segment_error,
    selected_segment_selection,
):
    if selected_segment_selection is None or selected_segment_error or not render_piano_rolls_checkbox.value:
        piano_roll_output = mo.md("")
    else:
        _view_data = cached_piano_roll_view_data(
            selected_segment_selection.segment,
            selected_segment_selection.duration_vocabulary,
        )
        piano_roll_output = piano_roll_chart_panel(
            _view_data,
            mo=mo,
            alt=alt,
            controls=segment_hand_controls,
        )
    piano_roll_output
    return


if __name__ == "__main__":
    app.run()
