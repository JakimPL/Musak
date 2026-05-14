from __future__ import annotations

import traceback
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from musak_model.data.config import SegmentationConfig
from musak_model.data.parser import parse_score
from musak_model.data.pipeline import segment_parsed_score
from musak_model.data.schema import ParsedScore, Segment

_PROCESSING_ERRORS = (Exception,)


class ProcessingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    path: Path
    parsed_score: ParsedScore | None = None
    segments: list[Segment]
    error_type: str | None = None
    error_message: str | None = None
    traceback_text: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_type is None


def process_score_safely(path: Path, *, window_bars: int, stride_bars: int) -> ProcessingResult:
    try:
        parsed_score = parse_score(path)
    except _PROCESSING_ERRORS as exception:
        return ProcessingResult(
            path=path,
            parsed_score=None,
            segments=[],
            error_type=type(exception).__name__,
            error_message=str(exception),
            traceback_text="".join(traceback.format_exception(exception)),
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
        )

    return ProcessingResult(path=path, parsed_score=parsed_score, segments=segments)
