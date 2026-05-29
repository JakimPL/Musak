from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Callable, Final, Literal, Protocol, cast

import torch
from torch import Tensor

from musak_model.conditioning.structural.schema import StructuralControlFeatures
from musak_model.conditioning.structural.vocabulary import StructuralControlVocabulary
from musak_model.conditioning.time_signature import TimeSignatureVocabulary
from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.decoder import segment_to_piano_roll_events
from musak_model.evaluation import diagnose_segment
from musak_model.evaluation.generation import ReferenceFreeGenerationMetric, reference_free_generation_metrics
from musak_model.generation.constraints import (
    GenerationConstraintError,
    GenerationConstraints,
    allowed_next_token_ids,
    mask_disallowed_logits,
)
from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import ModelConfig
from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.figure.counter import count_hand_figure_ngrams
from musak_model.n_grams.figure.parser import extract_hand_onset_runs
from musak_model.n_grams.figure.samples.schema import FigureNGramCountsByScale
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.n_grams.profile.io import read_figure_counts_for_groups
from musak_model.n_grams.profile.metrics.reference.distribution import figure_reference_alignment_metrics
from musak_model.n_grams.profile.rhythm.extraction import count_segment_rhythm_metrics
from musak_model.n_grams.profile.rhythm.io import read_rhythm_counts
from musak_model.n_grams.profile.rhythm.metrics import rhythm_reference_distribution_metrics
from musak_model.n_grams.profile.rhythm.schema import (
    RHYTHM_COUNTS_NAME,
    RHYTHM_DIR_NAME,
    RhythmCountCounter,
    RhythmCountKey,
    RhythmMetricKind,
)
from musak_model.paths import MODEL_CONFIG_DIR
from musak_model.tokens.config import TokenizationConfig
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import (
    BarToken,
    EndToken,
    Hand,
    HandToken,
    HoldToken,
    JoinWithPreviousToken,
    NoteToken,
    RestToken,
    ScaleType,
    StartToken,
    Token,
    scale_size_for_type,
)
from musak_model.tokens.text import tokens_from_text, tokens_to_text
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.conditioning import scale_type_to_id, time_signature_to_id
from musak_model.training.ingestion.schema import EncodedExercise

_FIGURE_PATTERN_MIN_N: Final = 1
_FIGURE_PATTERN_MAX_N: Final = 2
_REFERENCE_METRIC_PREFIX: Final = "notebook/figure_reference"
_RHYTHM_REFERENCE_METRIC_PREFIX: Final = "notebook/rhythm_reference"


class AutoregressiveModel(Protocol):
    def eval(self) -> AutoregressiveModel: ...

    def __call__(
        self,
        token_ids: Tensor,
        *,
        bar_positions: Tensor,
        difficulty_ids: Tensor | None = None,
        scale_type_ids: Tensor | None = None,
        time_signature_ids: Tensor | None = None,
        structural_control_ids: Tensor | None = None,
        token_padding_mask: Tensor | None = None,
    ) -> Tensor: ...


@dataclass(frozen=True)
class LoadedModel:
    model: HierarchicalAutoregressiveModel
    config: ModelConfig
    checkpoint_epoch: int | None
    best_validation_loss: float | None
    device: torch.device
    token_vocabulary: TokenVocabulary
    duration_vocabulary: DurationVocabulary


@dataclass(frozen=True)
class PromptData:
    tokens: list[Token]
    token_ids: list[int]
    model_input_ids: list[int]
    bar_positions: list[int]
    text: str


@dataclass(frozen=True)
class SamplingOptions:
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    greedy: bool = False
    seed: int | None = None
    constraints: GenerationConstraints | None = None
    scale_type: ScaleType | None = None
    time_signature: tuple[int, int] | None = None
    structural_features: StructuralControlFeatures | None = None


@dataclass(frozen=True)
class SampleTraceRow:
    step: int
    token_id: int
    token_text: str
    probability: float
    logit: float
    allowed_token_count: int | None


@dataclass(frozen=True)
class SamplingResult:
    tokens: list[Token]
    token_ids: list[int]
    new_token_ids: list[int]
    trace: list[SampleTraceRow]
    stop_reason: str
    reached_end: bool
    constraint_error: str | None

    @property
    def generated_token_count(self) -> int:
        return len(self.new_token_ids)


@dataclass(frozen=True)
class GeneratedOutput:
    sampling_result: SamplingResult
    decoded_segment: Segment
    decode_error: str | None
    duration_vocabulary: DurationVocabulary
    status_message: str
    status_kind: Literal["success", "warn"]


@dataclass(frozen=True)
class GenerationRequest:
    loaded_model: LoadedModel
    prompt_text: str
    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None
    greedy: bool
    seed: int
    scale_root: int
    scale_type: str
    time_numerator: int
    time_denominator: int
    target_bars: int
    use_constraints: bool
    minimum_duration: str
    allow_dotted: bool
    max_notes_per_hand: int | None
    max_onset_span: int | None
    max_gap: int | None
    max_span: int | None


def load_trained_model(
    checkpoint_path: Path,
    *,
    device: str,
    tokenization_config_path: Path | None = None,
    model_config_directory: Path | None = None,
) -> LoadedModel:
    resolved_device = torch.device(device)
    tokenization_config = (
        TokenizationConfig.load(tokenization_config_path)
        if tokenization_config_path is not None
        else TokenizationConfig.load()
    )
    duration_vocabulary = DurationVocabulary(tokenization_config)
    token_vocabulary = TokenVocabulary(duration_vocabulary)
    model_config = ModelConfig.load(
        vocabulary_size=token_vocabulary.vocabulary_size,
        config_directory=model_config_directory or MODEL_CONFIG_DIR,
    )
    model = HierarchicalAutoregressiveModel(model_config)
    state = cast(dict[str, object], torch.load(checkpoint_path, map_location=resolved_device))
    model.load_state_dict(cast(dict[str, Tensor], state["model_state_dict"]))
    model.to(resolved_device)
    model.eval()
    return LoadedModel(
        model=model,
        config=model_config,
        checkpoint_epoch=cast(int | None, state.get("epoch")),
        best_validation_loss=cast(float | None, state.get("best_validation_loss")),
        device=resolved_device,
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
    )


def empty_prompt(*, token_vocabulary: TokenVocabulary, duration_vocabulary: DurationVocabulary) -> PromptData:
    return prompt_from_tokens([], token_vocabulary=token_vocabulary, duration_vocabulary=duration_vocabulary)


def prompt_from_text(
    text: str,
    *,
    token_vocabulary: TokenVocabulary,
    duration_vocabulary: DurationVocabulary,
) -> PromptData:
    tokens: list[Token] = [
        token
        for token in tokens_from_text(text, duration_vocabulary=duration_vocabulary)
        if not isinstance(token, StartToken)
    ]
    return prompt_from_tokens(tokens, token_vocabulary=token_vocabulary, duration_vocabulary=duration_vocabulary)


def prompt_from_encoded_sample(
    sample: EncodedExercise,
    *,
    token_vocabulary: TokenVocabulary,
    duration_vocabulary: DurationVocabulary,
) -> PromptData:
    return prompt_from_tokens(
        token_vocabulary.decode(sample.token_ids),
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
    )


def prompt_from_tokens(
    tokens: list[Token],
    *,
    token_vocabulary: TokenVocabulary,
    duration_vocabulary: DurationVocabulary,
) -> PromptData:
    token_ids = token_vocabulary.encode(tokens)
    model_input_ids = [token_vocabulary.start_token_id, *token_ids]
    bar_positions = _model_bar_positions(tokens)
    return PromptData(
        tokens=tokens,
        token_ids=token_ids,
        model_input_ids=model_input_ids,
        bar_positions=bar_positions,
        text=tokens_to_text(tokens, duration_vocabulary=duration_vocabulary),
    )


def sample_autoregressive(
    model: AutoregressiveModel,
    prompt: PromptData,
    *,
    options: SamplingOptions,
    token_vocabulary: TokenVocabulary,
    duration_vocabulary: DurationVocabulary,
    model_config: ModelConfig | None = None,
    device: torch.device | None = None,
    progress_callback: Callable[[int, Token, str], None] | None = None,
) -> SamplingResult:
    resolved_device = torch.device("cpu") if device is None else device
    generator = torch.Generator(device=resolved_device)
    if options.seed is not None:
        generator.manual_seed(options.seed)

    model.eval()
    musical_token_ids = list(prompt.token_ids)
    model_input_ids = list(prompt.model_input_ids)
    trace: list[SampleTraceRow] = []
    constraint_error: str | None = None
    stop_reason = "max_tokens"

    with torch.no_grad():
        for step in range(options.max_new_tokens):
            if model_config is not None and len(model_input_ids) >= model_config.transformer.max_sequence_length:
                stop_reason = "max_sequence_length"
                break

            input_tensor = torch.tensor([model_input_ids], dtype=torch.long, device=resolved_device)
            bar_positions = torch.tensor(
                [_model_bar_positions_from_ids(musical_token_ids, token_vocabulary=token_vocabulary)],
                dtype=torch.long,
                device=resolved_device,
            )
            logits = model(
                input_tensor,
                bar_positions=bar_positions,
                scale_type_ids=_scale_type_tensor(options, model_config=model_config, device=resolved_device),
                time_signature_ids=_time_signature_tensor(options, model_config=model_config, device=resolved_device),
                structural_control_ids=_structural_control_tensor(
                    options, model_config=model_config, device=resolved_device
                ),
            )[0, -1]
            allowed_count: int | None = None
            if options.constraints is not None:
                try:
                    allowed_ids = allowed_next_token_ids(
                        musical_token_ids,
                        constraints=options.constraints,
                        token_vocabulary=token_vocabulary,
                        duration_vocabulary=duration_vocabulary,
                    )
                    allowed_count = len(allowed_ids)
                    logits = mask_disallowed_logits(logits, allowed_token_ids=allowed_ids)
                except GenerationConstraintError as exception:
                    constraint_error = str(exception)
                    stop_reason = "constraint_error"
                    break

            next_token_id, probability, selected_logit = _select_next_token_id(
                logits,
                options=options,
                generator=generator,
            )
            next_token = token_vocabulary.id_to_token(next_token_id)
            musical_token_ids.append(next_token_id)
            model_input_ids.append(next_token_id)
            stop_reason_for_step = "end_token" if isinstance(next_token, EndToken) else "running"
            if progress_callback is not None:
                progress_callback(step + 1, next_token, stop_reason_for_step)

            trace.append(
                SampleTraceRow(
                    step=step,
                    token_id=next_token_id,
                    token_text=next_token.to_text(duration_vocabulary=duration_vocabulary),
                    probability=probability,
                    logit=selected_logit,
                    allowed_token_count=allowed_count,
                )
            )
            if isinstance(next_token, EndToken):
                stop_reason = "end_token"
                break

    return SamplingResult(
        tokens=token_vocabulary.decode(musical_token_ids),
        token_ids=musical_token_ids,
        new_token_ids=musical_token_ids[len(prompt.token_ids) :],
        trace=trace,
        stop_reason=stop_reason,
        reached_end=stop_reason == "end_token",
        constraint_error=constraint_error,
    )


def sampling_result_to_segment(
    result: SamplingResult,
    *,
    scale_root: int,
    scale_type: ScaleType,
    time_numerator: int,
    time_denominator: int,
    source_file: Path = Path("generated"),
) -> Segment:
    return Segment(
        tokens=result.tokens,
        metadata=SegmentMetadata(
            scale_root=scale_root,
            scale_type=scale_type,
            time_numerator=time_numerator,
            time_denominator=time_denominator,
            bar_count=_display_bar_count(result.tokens),
            window_start_bar=0,
            source_file=source_file,
            difficulty_level=None,
        ),
    )


def trace_rows(result: SamplingResult) -> list[dict[str, float | int | str | None]]:
    return [
        {
            "step": row.step,
            "token_id": row.token_id,
            "token": row.token_text,
            "probability": row.probability,
            "logit": row.logit,
            "allowed": row.allowed_token_count,
        }
        for row in result.trace
    ]


def segment_decode_error(segment: Segment, *, duration_vocabulary: DurationVocabulary) -> str | None:
    try:
        segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary)
    except ValueError as exception:
        return str(exception)

    return None


def segment_event_count(segment: Segment, *, duration_vocabulary: DurationVocabulary) -> int | None:
    try:
        return len(segment_to_piano_roll_events(segment, duration_vocabulary=duration_vocabulary))
    except ValueError:
        return None


def segment_diagnostic_rows(segment: Segment, *, duration_vocabulary: DurationVocabulary) -> list[dict[str, object]]:
    diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)
    return [
        {"metric": "right silence", "value": _format_percent(diagnostics.right_silence_fraction)},
        {"metric": "left silence", "value": _format_percent(diagnostics.left_silence_fraction)},
        {"metric": "both hands silence", "value": _format_percent(diagnostics.both_hands_silence_fraction)},
        {"metric": "both hands active", "value": _format_percent(diagnostics.both_hands_active_fraction)},
        {"metric": "right only active", "value": _format_percent(diagnostics.right_only_active_fraction)},
        {"metric": "left only active", "value": _format_percent(diagnostics.left_only_active_fraction)},
        {"metric": "longest right silence", "value": f"{diagnostics.longest_right_silence_beats:.2f} beats"},
        {"metric": "longest left silence", "value": f"{diagnostics.longest_left_silence_beats:.2f} beats"},
        {
            "metric": "longest both-hands silence",
            "value": f"{diagnostics.longest_both_hands_silence_beats:.2f} beats",
        },
        {"metric": "right note onsets/bar", "value": f"{diagnostics.right_note_onsets_per_bar:.2f}"},
        {"metric": "left note onsets/bar", "value": f"{diagnostics.left_note_onsets_per_bar:.2f}"},
        {"metric": "silent bars", "value": diagnostics.silent_bar_count},
        {"metric": "silent bar share", "value": _format_percent(diagnostics.silent_bar_fraction)},
        {"metric": "silent edge bars", "value": diagnostics.silent_edge_bar_count},
        {"metric": "hand activity balance", "value": f"{diagnostics.hand_activity_balance:.3f}"},
        {"metric": "empty score", "value": diagnostics.empty_score},
        {"metric": "one hand only", "value": diagnostics.one_hand_only},
        {"metric": "note token share", "value": _format_percent(diagnostics.note_token_fraction)},
        {"metric": "rest token share", "value": _format_percent(diagnostics.rest_token_fraction)},
        {"metric": "hold token share", "value": _format_percent(diagnostics.hold_token_fraction)},
    ]


def generation_summary_metric_rows(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> list[dict[str, object]]:
    diagnostics = diagnose_segment(segment, duration_vocabulary=duration_vocabulary)
    metrics = reference_free_generation_metrics(diagnostics)
    return [{"metric": metric.label, "value": _format_reference_free_metric_value(metric)} for metric in metrics]


def _format_reference_free_metric_value(metric: ReferenceFreeGenerationMetric) -> object:
    match metric.kind:
        case "boolean":
            return metric.value
        case "count":
            return metric.value
        case "fraction":
            return f"{100 * float(metric.value):.1f}%"
        case "number":
            return f"{float(metric.value):.3f}"
        case "beats":
            return f"{float(metric.value):.2f} beats"


def figure_pattern_metric_rows(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
) -> list[dict[str, object]]:
    runs_by_hand = extract_hand_onset_runs(
        segment.tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=segment.time_numerator,
        time_denominator=segment.time_denominator,
    )
    counts_by_hand = count_hand_figure_ngrams(
        runs_by_hand,
        min_n=_FIGURE_PATTERN_MIN_N,
        max_n=_FIGURE_PATTERN_MAX_N,
        scale_size=scale_size_for_type(segment.scale_type),
    )
    single_onset_counts = _combined_figure_counts(counts_by_hand, n=1)
    two_onset_counts = _combined_figure_counts(counts_by_hand, n=2)
    single_onset_total = sum(single_onset_counts.values())
    two_onset_total = sum(two_onset_counts.values())
    chord_total = sum(count for figure, count in single_onset_counts.items() if not figure.monophonic)
    in_scale_total = sum(count for figure, count in single_onset_counts.items() if figure.in_scale)
    right_total = sum(counts_by_hand[Hand.RIGHT][1].values())
    left_total = sum(counts_by_hand[Hand.LEFT][1].values())
    return [
        _figure_metric_row(
            metric="single-onset figures",
            value=single_onset_total,
            description="Total note or chord onsets counted across both hands.",
        ),
        _figure_metric_row(
            metric="two-onset figures",
            value=two_onset_total,
            description="Total adjacent two-onset melodic or chord patterns within each hand.",
        ),
        _figure_metric_row(
            metric="unique single-onset figures",
            value=len(single_onset_counts),
            description="Distinct single-onset shapes after transposing each shape to its own first pitch.",
        ),
        _figure_metric_row(
            metric="unique two-onset figures",
            value=len(two_onset_counts),
            description="Distinct two-onset shapes after transposing each shape to its own first pitch.",
        ),
        _figure_metric_row(
            metric="single-onset variety",
            value=_format_optional_percent(len(single_onset_counts), single_onset_total),
            description="Distinct single-onset shapes divided by all single-onset figures.",
        ),
        _figure_metric_row(
            metric="chord figure share",
            value=_format_optional_percent(chord_total, single_onset_total),
            description="Single-onset figures containing more than one simultaneous note.",
        ),
        _figure_metric_row(
            metric="in-scale figure share",
            value=_format_optional_percent(in_scale_total, single_onset_total),
            description="Single-onset figures whose notes have no accidentals relative to the selected scale.",
        ),
        _figure_metric_row(
            metric="right-hand figure share",
            value=_format_optional_percent(right_total, single_onset_total),
            description="Single-onset figures that occur in the right hand.",
        ),
        _figure_metric_row(
            metric="left-hand figure share",
            value=_format_optional_percent(left_total, single_onset_total),
            description="Single-onset figures that occur in the left hand.",
        ),
    ]


@cache
def load_figure_reference_counts(
    path: Path,
    *,
    scale_type: ScaleType,
    groups: frozenset[tuple[Hand, int]],
) -> FigureNGramCountsByScale:
    return read_figure_counts_for_groups(path, scale_type=scale_type, groups=groups)


def figure_reference_count_groups(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    analysis_config: NGramAnalysisConfig | None = None,
) -> frozenset[tuple[Hand, int]]:
    resolved_config = analysis_config or NGramAnalysisConfig.load()
    generated_counts = _segment_figure_counts_by_scale(
        segment,
        duration_vocabulary=duration_vocabulary,
        min_n=resolved_config.min_n,
        max_n=resolved_config.max_n,
    )
    return frozenset(
        (hand, n)
        for counts_by_hand in generated_counts.values()
        for hand, counts_by_n in counts_by_hand.items()
        for n, figure_counts in counts_by_n.items()
        if figure_counts
    )


def figure_reference_alignment_metric_rows(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    reference_counts: FigureNGramCountsByScale,
    analysis_config: NGramAnalysisConfig | None = None,
) -> list[dict[str, object]]:
    resolved_config = analysis_config or NGramAnalysisConfig.load()
    generated_counts = _segment_figure_counts_by_scale(
        segment,
        duration_vocabulary=duration_vocabulary,
        min_n=resolved_config.min_n,
        max_n=resolved_config.max_n,
    )
    compatible_reference_counts = _compatible_reference_counts(
        reference_counts=reference_counts,
        generated_counts=generated_counts,
    )
    metrics = figure_reference_alignment_metrics(
        reference_counts=compatible_reference_counts,
        comparison_counts=generated_counts,
        metric_prefix=_REFERENCE_METRIC_PREFIX,
        common_mass_threshold=resolved_config.figure_common_mass_threshold,
    )
    return [
        _figure_metric_row(
            metric="reference groups compared",
            value=int(metrics[f"{_REFERENCE_METRIC_PREFIX}/count/distribution_groups"]),
            description="Reference slices with figure data, grouped by scale type, hand, and figure length.",
        ),
        _figure_metric_row(
            metric="figure identity distance",
            value=_format_float(metrics.get(f"{_REFERENCE_METRIC_PREFIX}/mean/identity_total_variation_distance")),
            description="How different the exact generated figure distribution is from the reference; 0 is closest.",
        ),
        _figure_metric_row(
            metric="contour distance",
            value=_format_float(metrics.get(f"{_REFERENCE_METRIC_PREFIX}/mean/contour_total_variation_distance")),
            description="How different the up, down, and repeated pitch-contour distribution is from the reference.",
        ),
        _figure_metric_row(
            metric="rhythm-shape distance",
            value=_format_float(
                metrics.get(f"{_REFERENCE_METRIC_PREFIX}/mean/duration_shape_total_variation_distance")
            ),
            description="How different the relative duration-pattern distribution is from the reference.",
        ),
        _figure_metric_row(
            metric="property distance",
            value=_format_float(metrics.get(f"{_REFERENCE_METRIC_PREFIX}/mean/property_total_variation_distance")),
            description="How different monophonic, chord-only, and in-scale figure properties are from the reference.",
        ),
        _figure_metric_row(
            metric="common figure mass",
            value=_format_percent_metric(metrics.get(f"{_REFERENCE_METRIC_PREFIX}/mean/common_figure_mass")),
            description="Share of generated figures that belong to the most common reference figures.",
        ),
        _figure_metric_row(
            metric="rare figure mass",
            value=_format_percent_metric(metrics.get(f"{_REFERENCE_METRIC_PREFIX}/mean/rare_figure_mass")),
            description="Share of generated figures found in the reference, but outside its common-figure set.",
        ),
        _figure_metric_row(
            metric="novel figure mass",
            value=_format_percent_metric(metrics.get(f"{_REFERENCE_METRIC_PREFIX}/mean/novel_figure_mass")),
            description="Share of generated figures that are absent from the matching reference slice.",
        ),
    ]


@cache
def load_rhythm_reference_counts(path: Path) -> RhythmCountCounter:
    return read_rhythm_counts(path)


def rhythm_reference_counts_path(figure_counts_path: Path) -> Path:
    return figure_counts_path.parent.parent / RHYTHM_DIR_NAME / RHYTHM_COUNTS_NAME


def rhythm_reference_alignment_metric_rows(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    reference_counts: RhythmCountCounter,
    analysis_config: NGramAnalysisConfig | None = None,
) -> list[dict[str, object]]:
    resolved_config = analysis_config or NGramAnalysisConfig.load()
    generated_counts = _segment_rhythm_counts(
        segment,
        duration_vocabulary=duration_vocabulary,
        analysis_config=resolved_config,
    )
    metrics = rhythm_reference_distribution_metrics(
        reference_counts=_compatible_rhythm_reference_counts(
            reference_counts=reference_counts,
            generated_counts=generated_counts,
        ),
        comparison_counts=generated_counts,
        metric_prefix=_RHYTHM_REFERENCE_METRIC_PREFIX,
    )
    return [
        _figure_metric_row(
            metric="rhythmic pattern distance",
            value=_format_float(
                metrics.get(f"{_RHYTHM_REFERENCE_METRIC_PREFIX}/mean/rhythm_ngram_total_variation_distance")
            ),
            description="How different onset-duration and inter-onset rhythm patterns are from the reference.",
        ),
        _figure_metric_row(
            metric="duration-value distance",
            value=_format_float(
                metrics.get(f"{_RHYTHM_REFERENCE_METRIC_PREFIX}/mean/duration_value_total_variation_distance")
            ),
            description="How different the generated onset-duration distribution is from the reference.",
        ),
        _figure_metric_row(
            metric="onset-grid distance",
            value=_format_float(
                metrics.get(f"{_RHYTHM_REFERENCE_METRIC_PREFIX}/mean/onset_grid_alignment_total_variation_distance")
            ),
            description="How different the generated onset grid-alignment distribution is from the reference.",
        ),
        _figure_metric_row(
            metric="duration-grid distance",
            value=_format_float(
                metrics.get(f"{_RHYTHM_REFERENCE_METRIC_PREFIX}/mean/duration_grid_alignment_total_variation_distance")
            ),
            description="How different the generated duration grid-alignment distribution is from the reference.",
        ),
        _figure_metric_row(
            metric="duration entropy difference",
            value=_format_float(metrics.get(f"{_RHYTHM_REFERENCE_METRIC_PREFIX}/mean/duration_entropy_absolute_error")),
            description="Absolute difference between generated and reference duration-distribution entropy.",
        ),
        _figure_metric_row(
            metric="strong-beat share difference",
            value=_format_float(
                metrics.get(f"{_RHYTHM_REFERENCE_METRIC_PREFIX}/mean/strong_beat_onset_fraction_absolute_error")
            ),
            description="Absolute difference between generated and reference strong-beat onset share.",
        ),
    ]


def rhythm_grid_metric_rows(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    analysis_config: NGramAnalysisConfig | None = None,
) -> list[dict[str, object]]:
    resolved_config = analysis_config or NGramAnalysisConfig.load()
    grid_denominator = max(resolved_config.grid_alignment_denominators)
    rhythm_counts = _segment_rhythm_counts(
        segment,
        duration_vocabulary=duration_vocabulary,
        analysis_config=resolved_config,
    )
    onset_grid_count = _rhythm_count(
        rhythm_counts,
        segment=segment,
        kind="onset_grid_alignment",
        parameter=str(grid_denominator),
        value="aligned",
    )
    duration_grid_count = _rhythm_count(
        rhythm_counts,
        segment=segment,
        kind="duration_grid_alignment",
        parameter=str(grid_denominator),
        value="aligned",
    )
    strong_beat_count = _rhythm_count(
        rhythm_counts,
        segment=segment,
        kind="strong_beat_onset",
        parameter="",
        value="strong",
    )
    onset_total = _rhythm_kind_total(
        rhythm_counts,
        segment=segment,
        kind="strong_beat_onset",
        parameter="",
    )
    return [
        _figure_metric_row(
            metric="rhythmic onsets",
            value=onset_total,
            description="Total note or chord onsets used for rhythm-grid checks, counted per hand.",
        ),
        _figure_metric_row(
            metric=f"onset grid fit (1/{grid_denominator})",
            value=_format_optional_percent(onset_grid_count, onset_total),
            description="Share of onsets that start exactly on the configured rhythmic grid.",
        ),
        _figure_metric_row(
            metric=f"duration grid fit (1/{grid_denominator})",
            value=_format_optional_percent(duration_grid_count, onset_total),
            description="Share of onset durations that fit exactly on the configured rhythmic grid.",
        ),
        _figure_metric_row(
            metric="strong-beat onset share",
            value=_format_optional_percent(strong_beat_count, onset_total),
            description="Share of onsets that begin on configured strong-beat offsets within the bar.",
        ),
    ]


def _figure_metric_row(*, metric: str, value: object, description: str) -> dict[str, object]:
    return {"metric": metric, "value": value, "description": description}


def _segment_figure_counts_by_scale(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    min_n: int,
    max_n: int,
) -> FigureNGramCountsByScale:
    runs_by_hand = extract_hand_onset_runs(
        segment.tokens,
        duration_vocabulary=duration_vocabulary,
        time_numerator=segment.time_numerator,
        time_denominator=segment.time_denominator,
    )
    counts_by_hand = count_hand_figure_ngrams(
        runs_by_hand,
        min_n=min_n,
        max_n=max_n,
        scale_size=scale_size_for_type(segment.scale_type),
    )
    return {segment.scale_type: counts_by_hand}


def _compatible_reference_counts(
    *,
    reference_counts: FigureNGramCountsByScale,
    generated_counts: FigureNGramCountsByScale,
) -> FigureNGramCountsByScale:
    compatible_counts: FigureNGramCountsByScale = {}
    for scale_type, generated_counts_by_hand in generated_counts.items():
        reference_counts_by_hand = reference_counts.get(scale_type)
        if reference_counts_by_hand is None:
            continue

        for hand, generated_counts_by_n in generated_counts_by_hand.items():
            reference_counts_by_n = reference_counts_by_hand.get(hand)
            if reference_counts_by_n is None:
                continue

            for n, generated_figure_counts in generated_counts_by_n.items():
                if not generated_figure_counts:
                    continue

                reference_figure_counts = reference_counts_by_n.get(n)
                if not reference_figure_counts:
                    continue

                compatible_counts.setdefault(scale_type, {}).setdefault(hand, {})[n] = reference_figure_counts

    return compatible_counts


def _segment_rhythm_counts(
    segment: Segment,
    *,
    duration_vocabulary: DurationVocabulary,
    analysis_config: NGramAnalysisConfig,
) -> RhythmCountCounter:
    return count_segment_rhythm_metrics(
        segment,
        duration_vocabulary=duration_vocabulary,
        rhythm_min_n=analysis_config.rhythm_min_n,
        rhythm_max_n=analysis_config.rhythm_max_n,
        grid_alignment_denominators=analysis_config.grid_alignment_denominators,
        strong_beat_offsets=analysis_config.strong_beat_offsets,
    )


def _compatible_rhythm_reference_counts(
    *,
    reference_counts: RhythmCountCounter,
    generated_counts: RhythmCountCounter,
) -> RhythmCountCounter:
    generated_groups = {
        (key.scale_type, key.time_signature, key.hand, key.kind, key.parameter) for key in generated_counts
    }
    return Counter(
        {
            key: count
            for key, count in reference_counts.items()
            if (key.scale_type, key.time_signature, key.hand, key.kind, key.parameter) in generated_groups
        }
    )


def _rhythm_count(
    counts: RhythmCountCounter,
    *,
    segment: Segment,
    kind: RhythmMetricKind,
    parameter: str,
    value: str,
) -> int:
    time_signature = f"{segment.time_numerator}/{segment.time_denominator}"
    return sum(
        counts[
            RhythmCountKey(
                scale_type=segment.scale_type.value,
                time_signature=time_signature,
                hand=hand.value,
                kind=kind,
                parameter=parameter,
                value=value,
            )
        ]
        for hand in Hand
    )


def _rhythm_kind_total(
    counts: RhythmCountCounter,
    *,
    segment: Segment,
    kind: RhythmMetricKind,
    parameter: str,
) -> int:
    time_signature = f"{segment.time_numerator}/{segment.time_denominator}"
    return sum(
        count
        for key, count in counts.items()
        if key.scale_type == segment.scale_type.value
        and key.time_signature == time_signature
        and key.kind == kind
        and key.parameter == parameter
    )


def _combined_figure_counts(
    counts_by_hand: dict[Hand, dict[int, Counter[FigureNGram]]],
    *,
    n: int,
) -> Counter[FigureNGram]:
    combined: Counter[FigureNGram] = Counter()
    for counts_by_n in counts_by_hand.values():
        combined.update(counts_by_n[n])

    return combined


def _format_optional_percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "-"

    return _format_percent(numerator / denominator)


def _format_percent_metric(value: float | None) -> str:
    return "-" if value is None else _format_percent(value)


def _format_percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _display_bar_count(tokens: list[Token]) -> int:
    completed_bars = sum(1 for token in tokens if isinstance(token, BarToken))
    trailing_tokens_after_bar = False
    for token in reversed(tokens):
        if isinstance(token, EndToken):
            continue
        if isinstance(token, BarToken):
            break
        if isinstance(token, (HandToken, JoinWithPreviousToken, NoteToken, RestToken, HoldToken)):
            trailing_tokens_after_bar = True
            break

    return completed_bars + int(trailing_tokens_after_bar)


def _model_bar_positions(tokens: list[Token]) -> list[int]:
    positions = [0]
    bar_index = 0
    for token in tokens:
        positions.append(bar_index)
        if isinstance(token, BarToken):
            bar_index += 1

    return positions


def _model_bar_positions_from_ids(token_ids: list[int], *, token_vocabulary: TokenVocabulary) -> list[int]:
    return _model_bar_positions(token_vocabulary.decode(token_ids))


def _select_next_token_id(
    logits: Tensor,
    *,
    options: SamplingOptions,
    generator: torch.Generator,
) -> tuple[int, float, float]:
    if options.greedy:
        token_id = int(torch.argmax(logits).item())
        probabilities = torch.softmax(logits, dim=-1)
        return token_id, float(probabilities[token_id].item()), float(logits[token_id].item())

    filtered_logits = _apply_sampling_filters(logits, options=options)
    probabilities = torch.softmax(filtered_logits / max(options.temperature, 1e-6), dim=-1)
    token_id = int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())
    return token_id, float(probabilities[token_id].item()), float(logits[token_id].item())


def _apply_sampling_filters(logits: Tensor, *, options: SamplingOptions) -> Tensor:
    filtered = logits.clone()
    if options.top_k is not None and options.top_k > 0 and options.top_k < filtered.numel():
        threshold = torch.topk(filtered, options.top_k).values[-1]
        filtered = torch.where(filtered < threshold, torch.full_like(filtered, float("-inf")), filtered)

    if options.top_p is not None and 0 < options.top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative > options.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[sorted_indices[remove]] = float("-inf")

    return filtered


def _scale_type_tensor(
    options: SamplingOptions,
    *,
    model_config: ModelConfig | None,
    device: torch.device,
) -> Tensor | None:
    if model_config is None or options.scale_type is None:
        return None

    return torch.tensor([scale_type_to_id(options.scale_type)], dtype=torch.long, device=device)


def _time_signature_tensor(
    options: SamplingOptions,
    *,
    model_config: ModelConfig | None,
    device: torch.device,
) -> Tensor | None:
    if model_config is None or options.time_signature is None:
        return None

    vocabulary = TimeSignatureVocabulary(model_config.conditioning.time_signature)
    return torch.tensor(
        [time_signature_to_id(options.time_signature, vocabulary=vocabulary)],
        dtype=torch.long,
        device=device,
    )


def _structural_control_tensor(
    options: SamplingOptions,
    *,
    model_config: ModelConfig | None,
    device: torch.device,
) -> Tensor | None:
    if model_config is None or options.structural_features is None:
        return None

    vocabulary = StructuralControlVocabulary(model_config.conditioning.structural)
    return torch.tensor([vocabulary.features_to_ids(options.structural_features)], dtype=torch.long, device=device)
