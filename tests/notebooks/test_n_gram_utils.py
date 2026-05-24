from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.analysis.n_grams import FigureNGram
from notebooks.utils.n_grams import (
    FIGURE_PERCENT_COLUMN,
    analysis_result_files,
    figure_display_unit,
    figure_group_summary,
    figure_ngram_to_score_data,
    parse_figure_ngram,
    read_figure_count_frame,
    top_figure_frame,
)


def test_read_figure_count_frame_parses_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "figures.csv"
    path.write_text(
        "\n".join(
            [
                "scale_type,hand,n,count,figure",
                'major,right,2,3,"{""onsets"":[[[[0,0]],""1""],[[[1,0]],""1""]]}"',
            ]
        ),
        encoding="utf-8",
    )

    frame = read_figure_count_frame(path)

    assert frame["n"].tolist() == [2]
    assert frame["count"].tolist() == [3]


def test_read_figure_count_frame_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "figures.csv"
    path.write_text("scale_type,hand,n,count\nmajor,right,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="figure"):
        read_figure_count_frame(path)


def test_figure_group_summary_counts_unique_and_total_figures(tmp_path: Path) -> None:
    frame = read_figure_count_frame(_write_count_csv(tmp_path))

    summary = figure_group_summary(frame)

    row = summary.iloc[0]
    assert int(row["total_count"]) == 5
    assert int(row["unique_figures"]) == 2


def test_top_figure_frame_filters_and_adds_percent(tmp_path: Path) -> None:
    frame = read_figure_count_frame(_write_count_csv(tmp_path))

    top = top_figure_frame(frame, scale_type="major", hand="right", n=2, top_n=1)

    assert top["count"].tolist() == [3]
    assert top[FIGURE_PERCENT_COLUMN].tolist() == [0.6]


def test_parse_figure_ngram_reads_exported_json() -> None:
    figure = parse_figure_ngram('{"onsets":[[[[0,0]],"1"],[[[1,0]],"2"]]}')

    assert figure.onsets[1][1] == Fraction(2)


def test_figure_ngram_to_score_data_renders_relative_degrees_and_durations() -> None:
    figure = FigureNGram(onsets=((((0, 0), (2, 1)), Fraction(1)), (((7, -1),), Fraction(2))))

    score = figure_ngram_to_score_data(figure, preferred_unit=Fraction(1, 8))

    notes = score.rows[0][0].voices[0].notes
    assert notes[0].keys == ["c/4", "e/4"]
    assert notes[0].accidentals == [None, "#"]
    assert notes[0].duration == "8"
    assert notes[1].keys == ["c/5"]
    assert notes[1].accidentals == ["b"]
    assert notes[1].duration == "q"


def test_figure_display_unit_uses_smaller_unit_when_preferred_is_too_large() -> None:
    figure = FigureNGram(onsets=((((0, 0),), Fraction(8)),))

    assert figure_display_unit(figure, preferred_unit=Fraction(1, 4)) == Fraction(1, 8)


def test_analysis_result_files_lists_csv_files(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("", encoding="utf-8")
    (tmp_path / "b.txt").write_text("", encoding="utf-8")

    assert analysis_result_files(tmp_path) == [tmp_path / "a.csv"]


def _write_count_csv(tmp_path: Path) -> Path:
    path = tmp_path / "figures.csv"
    path.write_text(
        "\n".join(
            [
                "scale_type,hand,n,count,figure",
                'major,right,2,3,"{""onsets"":[[[[0,0]],""1""],[[[1,0]],""1""]]}"',
                'major,right,2,2,"{""onsets"":[[[[0,0]],""1""],[[[2,0]],""1""]]}"',
            ]
        ),
        encoding="utf-8",
    )
    return path
