from dataclasses import dataclass
from typing import Literal

from musak_model.evaluation.diagnostics import SegmentDiagnostics

type ReferenceFreeMetricKind = Literal["boolean", "count", "fraction", "number", "beats"]
type ReferenceFreeMetricValue = bool | float | int


@dataclass(frozen=True)
class ReferenceFreeGenerationMetric:
    key: str
    label: str
    value: ReferenceFreeMetricValue
    kind: ReferenceFreeMetricKind


def reference_free_generation_metrics(diagnostics: SegmentDiagnostics) -> list[ReferenceFreeGenerationMetric]:
    return [
        ReferenceFreeGenerationMetric(
            key="empty_score",
            label="empty score",
            value=diagnostics.empty_score,
            kind="boolean",
        ),
        ReferenceFreeGenerationMetric(
            key="one_hand_only",
            label="one hand only",
            value=diagnostics.one_hand_only,
            kind="boolean",
        ),
        ReferenceFreeGenerationMetric(
            key="both_hands_active_fraction",
            label="both hands active",
            value=diagnostics.both_hands_active_fraction,
            kind="fraction",
        ),
        ReferenceFreeGenerationMetric(
            key="hand_activity_balance",
            label="hand activity balance",
            value=diagnostics.hand_activity_balance,
            kind="number",
        ),
        ReferenceFreeGenerationMetric(
            key="silent_bar_fraction",
            label="silent bar share",
            value=diagnostics.silent_bar_fraction,
            kind="fraction",
        ),
        ReferenceFreeGenerationMetric(
            key="in_scale_note_fraction",
            label="in-scale note share",
            value=diagnostics.in_scale_note_fraction,
            kind="fraction",
        ),
        ReferenceFreeGenerationMetric(
            key="note_density_per_beat",
            label="note density / beat",
            value=diagnostics.note_density_per_beat,
            kind="number",
        ),
        ReferenceFreeGenerationMetric(
            key="onset_density_per_beat",
            label="onset density / beat",
            value=diagnostics.onset_density_per_beat,
            kind="number",
        ),
        ReferenceFreeGenerationMetric(
            key="shortest_note_duration_beats",
            label="shortest note duration",
            value=diagnostics.shortest_note_duration_beats,
            kind="beats",
        ),
        ReferenceFreeGenerationMetric(
            key="has_dotted_notes",
            label="has dotted notes",
            value=diagnostics.has_dotted_notes,
            kind="boolean",
        ),
        ReferenceFreeGenerationMetric(
            key="max_notes_per_onset",
            label="max notes / onset",
            value=diagnostics.max_notes_per_onset,
            kind="count",
        ),
        ReferenceFreeGenerationMetric(
            key="max_onset_span_semitones",
            label="max onset span",
            value=diagnostics.max_onset_span_semitones,
            kind="count",
        ),
        ReferenceFreeGenerationMetric(
            key="max_melodic_gap_semitones",
            label="max melodic gap",
            value=diagnostics.max_melodic_gap_semitones,
            kind="count",
        ),
        ReferenceFreeGenerationMetric(
            key="synchronized_onset_fraction",
            label="synchronized onset share",
            value=diagnostics.synchronized_onset_fraction,
            kind="fraction",
        ),
    ]
