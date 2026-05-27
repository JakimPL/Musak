import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from random import Random

    import marimo as mo

    from musak_model.paths import DEFAULT_TRAINING_FIGURE_DIR
    from musak_model.synthetic import load_figure_vocabulary
    from musak_model.tokens.schema import Hand, ScaleType
    from notebooks.utils import selected_file

    return DEFAULT_TRAINING_FIGURE_DIR, Hand, Random, ScaleType, load_figure_vocabulary, mo, selected_file


@app.cell
def _(mo):
    mo.md("# Synthetic Sight-Reading Generator")
    return


@app.cell
def _(DEFAULT_TRAINING_FIGURE_DIR, mo):
    vocabulary_browser = mo.ui.file_browser(
        initial_path=DEFAULT_TRAINING_FIGURE_DIR if DEFAULT_TRAINING_FIGURE_DIR.exists() else ".",
        filetypes=[".csv"],
        selection_mode="file",
        multiple=False,
        label="Figure vocabulary counts CSV",
    )
    vocabulary_browser
    return (vocabulary_browser,)


@app.cell
def _(load_figure_vocabulary, mo, selected_file, vocabulary_browser):
    vocabulary = None
    vocabulary_selection = (
        selected_file(
            vocabulary_browser,
            supported_suffixes=frozenset({".csv"}),
            description="figure vocabulary counts CSV",
        )
        if vocabulary_browser.value
        else None
    )
    if vocabulary_selection is None:
        vocabulary_status = mo.callout("Select a figure vocabulary counts CSV.", kind="warn")
    elif vocabulary_selection.path is None:
        vocabulary_status = mo.callout(
            vocabulary_selection.message or "Figure vocabulary CSV is unavailable.",
            kind="warn",
        )
    else:
        try:
            vocabulary = load_figure_vocabulary(vocabulary_selection.path)
        except (FileNotFoundError, ValueError) as exception:
            vocabulary_status = mo.callout(f"Figure vocabulary could not be loaded: {exception}", kind="warn")
        else:
            vocabulary_status = mo.callout(
                f"Loaded `{vocabulary_selection.path.name}`: "
                f"{vocabulary.unique_count} figures, {vocabulary.total_count} occurrences.",
                kind="success",
            )

    vocabulary_status
    return (vocabulary,)


@app.cell
def _(Hand, Random, ScaleType, mo, vocabulary):
    figure = None
    if vocabulary is None:
        preview = mo.md("")
    else:
        right_hand_figures = vocabulary.filter(
            scale_type=ScaleType.MAJOR,
            hand=Hand.RIGHT,
            n=2,
            in_scale=True,
        )
        if right_hand_figures.unique_count == 0:
            preview = mo.callout("No matching right-hand major 2-gram figures found.", kind="warn")
        else:
            entry = right_hand_figures.sample(rng=Random(123), commonness_bias=1.0)
            figure = entry.figure
            preview = mo.md(f"Sample figure: `{figure}`")

    preview
    return (figure,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
