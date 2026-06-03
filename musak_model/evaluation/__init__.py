from typing import TYPE_CHECKING

from musak_model.evaluation.diagnostics import SegmentDiagnostics, diagnose_segment

if TYPE_CHECKING:
    from musak_model.evaluation.generation import GenerationSuiteEvaluator

__all__ = ["GenerationSuiteEvaluator", "SegmentDiagnostics", "diagnose_segment"]


def __getattr__(name: str) -> object:
    if name == "GenerationSuiteEvaluator":
        from musak_model.evaluation.generation import GenerationSuiteEvaluator

        return GenerationSuiteEvaluator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
