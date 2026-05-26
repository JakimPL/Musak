from pathlib import Path
from zipfile import ZipFile

from musak_model.processing.parser.title import score_title


def test_score_title_returns_musicxml_movement_title(tmp_path: Path) -> None:
    source_path = tmp_path / "piece.musicxml"
    source_path.write_text(
        """
        <score-partwise>
          <movement-title>Prelude</movement-title>
        </score-partwise>
        """,
        encoding="utf-8",
    )

    assert score_title(source_path) == "Prelude"


def test_score_title_returns_musicxml_work_title(tmp_path: Path) -> None:
    source_path = tmp_path / "piece.musicxml"
    source_path.write_text(
        """
        <score-partwise>
          <work>
            <work-title>Notebook Sketch</work-title>
          </work>
        </score-partwise>
        """,
        encoding="utf-8",
    )

    assert score_title(source_path) == "Notebook Sketch"


def test_score_title_returns_compressed_musicxml_title(tmp_path: Path) -> None:
    source_path = tmp_path / "piece.mxl"
    with ZipFile(source_path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            """
            <container>
              <rootfiles>
                <rootfile full-path="score.musicxml" />
              </rootfiles>
            </container>
            """,
        )
        archive.writestr(
            "score.musicxml",
            """
            <score-partwise>
              <movement-title>Compressed Piece</movement-title>
            </score-partwise>
            """,
        )

    assert score_title(source_path) == "Compressed Piece"


def test_score_title_returns_empty_for_malformed_musicxml(tmp_path: Path) -> None:
    source_path = tmp_path / "piece.musicxml"
    source_path.write_text("<score-partwise>")

    assert score_title(source_path) == ""
