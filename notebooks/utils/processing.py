from __future__ import annotations

import traceback
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

from music21.exceptions21 import Music21Exception
from pydantic import BaseModel, ConfigDict, ValidationError

from musak_model.data.cleaning import clean_parsed_score
from musak_model.data.config import SegmentationConfig
from musak_model.data.parser import parse_score
from musak_model.data.pipeline import segment_parsed_score
from musak_model.data.schema import ParsedScore, Segment
from musak_model.processing.diagnostics import ParseDiagnosticsCapture
from musak_model.processing.manifest import ParsedManifestField, read_parsed_manifest

_PROCESSING_ERRORS: tuple[type[Exception], ...] = (
    Music21Exception,
    OSError,
    ParseError,
    BadZipFile,
    TypeError,
    ValueError,
    ValidationError,
)


class ProcessingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    path: Path
    parsed_score: ParsedScore | None = None
    segments: list[Segment]
    error_type: str | None = None
    error_message: str | None = None
    traceback_text: str | None = None
    parse_diagnostics: str = ""

    @property
    def succeeded(self) -> bool:
        return self.error_type is None


def process_score_safely(
    path: Path,
    *,
    window_bars: int,
    stride_bars: int,
) -> ProcessingResult:
    parse_diagnostics = ""
    captured_diagnostics: ParseDiagnosticsCapture | None = None
    try:
        with ParseDiagnosticsCapture() as captured_diagnostics:
            parsed_score = clean_parsed_score(parse_score(path))
        parse_diagnostics = captured_diagnostics.text()
    except _PROCESSING_ERRORS as exception:
        if captured_diagnostics is not None:
            parse_diagnostics = captured_diagnostics.text()
        return ProcessingResult(
            path=path,
            parsed_score=None,
            segments=[],
            error_type=type(exception).__name__,
            error_message=str(exception),
            traceback_text="".join(traceback.format_exception(exception)),
            parse_diagnostics=parse_diagnostics,
        )

    try:
        segments = segment_parsed_score(
            parsed_score,
            path,
            segmentation=SegmentationConfig(window_bars=window_bars, stride_bars=stride_bars),
        )
    except _PROCESSING_ERRORS as exception:
        return ProcessingResult(
            path=path,
            parsed_score=parsed_score,
            segments=[],
            error_type=type(exception).__name__,
            error_message=str(exception),
            traceback_text="".join(traceback.format_exception(exception)),
            parse_diagnostics=parse_diagnostics,
        )

    return ProcessingResult(
        path=path,
        parsed_score=parsed_score,
        segments=segments,
        parse_diagnostics=parse_diagnostics,
    )


def segment_parsed_score_safely(
    parsed_score: ParsedScore,
    path: Path,
    *,
    window_bars: int,
    stride_bars: int,
    parse_diagnostics: str = "",
) -> ProcessingResult:
    try:
        segments = segment_parsed_score(
            parsed_score,
            path,
            segmentation=SegmentationConfig(window_bars=window_bars, stride_bars=stride_bars),
        )
    except _PROCESSING_ERRORS as exception:
        return ProcessingResult(
            path=path,
            parsed_score=parsed_score,
            segments=[],
            error_type=type(exception).__name__,
            error_message=str(exception),
            traceback_text="".join(traceback.format_exception(exception)),
            parse_diagnostics=parse_diagnostics,
        )

    return ProcessingResult(
        path=path,
        parsed_score=parsed_score,
        segments=segments,
        parse_diagnostics=parse_diagnostics,
    )


def encoded_segments_result(path: Path, *, segments: list[Segment]) -> ProcessingResult:
    return ProcessingResult(path=path, parsed_score=None, segments=segments)


def parsed_score_manifest_diagnostics(path: Path) -> str:
    manifest_path = _nearest_parsed_manifest(path)
    if manifest_path is None:
        return ""

    relative_path = path.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
    for row in read_parsed_manifest(manifest_path):
        if row.get(ParsedManifestField.PARSED_PATH, "") == relative_path:
            return row.get(ParsedManifestField.PARSE_DIAGNOSTICS, "")

    return ""


def _nearest_parsed_manifest(path: Path) -> Path | None:
    for parent in path.resolve().parents:
        manifest_path = parent / "parsed.csv"
        if manifest_path.exists():
            return manifest_path

    return None
