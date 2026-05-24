from fractions import Fraction
from pathlib import Path

import pytest

from musak_model.analysis.n_grams import FigureNGram
from notebooks.utils.n_grams import (
    FIGURE_CHORDS_ONLY_COLUMN,
    FIGURE_IN_SCALE_COLUMN,
    FIGURE_MONOPHONIC_COLUMN,
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
    assert top[FIGURE_TEXT_COLUMN].tolist() == ["0(1) +1(1)"]


def test_figure_filter_frame_returns_all_filtered_rows(tmp_path: Path) -> None:
    frame = read_figure_count_frame(_write_count_csv(tmp_path))

    filtered = figure_filter_frame(frame, scale_type="major", hand="right", n=2)

    assert filtered["count"].tolist() == [3, 2]
    assert filtered[FIGURE_TEXT_COLUMN].tolist() == ["0(1) +1(1)", "0(1) +2(1)"]


def test_figure_filter_frame_accepts_all_n_values(tmp_path: Path) -> None:
    frame = read_figure_count_frame(_write_mixed_n_count_csv(tmp_path))

    filtered = figure_filter_frame(frame, scale_type="major", hand="right", n=None)

    assert filtered["n"].tolist() == [3, 2]
    assert filtered["count"].tolist() == [5, 3]
    assert filtered[FIGURE_TEXT_COLUMN].tolist() == ["0(1) +1(1) +2(1)", "0(1) +1(1)"]


def test_top_figure_frame_accepts_all_n_values(tmp_path: Path) -> None:
    frame = read_figure_count_frame(_write_mixed_n_count_csv(tmp_path))

    top = top_figure_frame(frame, scale_type="major", hand="right", n=None, top_n=1)

    assert top["n"].tolist() == [3]
    assert top[FIGURE_TEXT_COLUMN].tolist() == ["0(1) +1(1) +2(1)"]


def test_figure_filter_frame_adds_figure_properties(tmp_path: Path) -> None:
    frame = read_figure_count_frame(_write_count_csv(tmp_path))

    filtered = figure_filter_frame(frame, scale_type="major", hand="right", n=2)

    assert filtered[FIGURE_MONOPHONIC_COLUMN].tolist() == [True, True]
    assert filtered[FIGURE_CHORDS_ONLY_COLUMN].tolist() == [False, False]
    assert filtered[FIGURE_IN_SCALE_COLUMN].tolist() == [True, True]


def test_figure_property_distribution_weights_by_count(tmp_path: Path) -> None:
    frame = read_figure_count_frame(_write_property_count_csv(tmp_path))

    distribution = figure_property_distribution(frame)

    lookup = {
        (str(row[FIGURE_PROPERTY_COLUMN]), bool(row[FIGURE_PROPERTY_VALUE_COLUMN])): (
            int(row["count"]),
            float(row[FIGURE_PERCENT_COLUMN]),
        )
        for _, row in distribution.iterrows()
    }
    assert lookup[(FIGURE_MONOPHONIC_COLUMN, True)] == (3, 0.3)
    assert lookup[(FIGURE_MONOPHONIC_COLUMN, False)] == (7, 0.7)
    assert lookup[(FIGURE_CHORDS_ONLY_COLUMN, True)] == (5, 0.5)
    assert lookup[(FIGURE_CHORDS_ONLY_COLUMN, False)] == (5, 0.5)
    assert lookup[(FIGURE_IN_SCALE_COLUMN, True)] == (5, 0.5)
    assert lookup[(FIGURE_IN_SCALE_COLUMN, False)] == (5, 0.5)


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


def _write_mixed_n_count_csv(tmp_path: Path) -> Path:
    path = tmp_path / "mixed-n.csv"
    path.write_text(
        "\n".join(
            [
                "scale_type,hand,n,count,figure",
                'major,right,2,3,"{""onsets"":[[[[0,0]],""1""],[[[1,0]],""1""]]}"',
                'major,right,3,5,"{""onsets"":[[[[0,0]],""1""],[[[1,0]],""1""],[[[2,0]],""1""]]}"',
                'major,left,2,11,"{""onsets"":[[[[0,0]],""1""],[[[-1,0]],""1""]]}"',
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_property_count_csv(tmp_path: Path) -> Path:
    path = tmp_path / "properties.csv"
    path.write_text(
        "\n".join(
            [
                "scale_type,hand,n,count,figure",
                'major,right,2,3,"{""onsets"":[[[[0,0]],""1""],[[[1,0]],""1""]]}"',
                'major,right,2,5,"{""onsets"":[[[[0,0],[2,1]],""1""],[[[4,0],[6,0]],""1""]]}"',
                'major,right,2,2,"{""onsets"":[[[[0,0],[2,0]],""1""],[[[1,0]],""1""]]}"',
            ]
        ),
        encoding="utf-8",
    )
    return path
