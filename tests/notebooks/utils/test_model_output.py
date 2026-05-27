from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.n_grams.config import NGramAnalysisConfig
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import EndToken, Hand, HandToken, HoldToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise
from notebooks.utils.model_output import (
    SamplingOptions,
    SamplingResult,
    empty_prompt,
    figure_pattern_metric_rows,
    figure_reference_alignment_metric_rows,
    generation_summary_metric_rows,
    prompt_from_encoded_sample,
    prompt_from_text,
    rhythm_grid_metric_rows,
    sample_autoregressive,
    sampling_result_to_segment,
    segment_decode_error,
    segment_event_count,
)


class LogitStreamModel(nn.Module):
    def __init__(self, logits: list[Tensor]) -> None:
        super().__init__()
        self._logits = logits
        self._call_index = 0

    def forward(
        self,
        token_ids: Tensor,
        *,
        bar_positions: Tensor,
        difficulty_ids: Tensor | None = None,
        scale_type_ids: Tensor | None = None,
        time_signature_ids: Tensor | None = None,
        structural_control_ids: Tensor | None = None,
        token_padding_mask: Tensor | None = None,
    ) -> Tensor:
        row = self._logits[self._call_index]
        self._call_index += 1
        output = torch.zeros(
            token_ids.size(0),
            token_ids.size(1),
            row.size(0),
            dtype=row.dtype,
            device=token_ids.device,
        )
        output[:, -1, :] = row
        return output


def test_sampler_uses_mock_logit_stream(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    note = NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id)
    note_id = token_vocabulary.token_to_id(note)
    first_logits = torch.full((token_vocabulary.vocabulary_size,), -10.0)
    first_logits[note_id] = 10.0
    second_logits = torch.full((token_vocabulary.vocabulary_size,), -10.0)
    second_logits[token_vocabulary.end_token_id] = 10.0
    model = LogitStreamModel([first_logits, second_logits])

    result = sample_autoregressive(
        model,
        empty_prompt(token_vocabulary=token_vocabulary, duration_vocabulary=duration_vocabulary),
        options=SamplingOptions(max_new_tokens=4, greedy=True),
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
    )

    assert result.stop_reason == "end_token"
    assert result.new_token_ids == [note_id, token_vocabulary.end_token_id]
    assert result.tokens[-1] == EndToken()


def test_prompt_from_text_decodes_tokens_and_prepends_start_for_model_input(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    prompt = prompt_from_text(
        "R 1(1:4)",
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
    )

    assert prompt.tokens[0] == HandToken(hand=Hand.RIGHT)
    assert prompt.model_input_ids[0] == token_vocabulary.start_token_id
    assert len(prompt.bar_positions) == len(prompt.model_input_ids)


def test_prompt_from_encoded_sample_uses_sample_ids(
    duration_vocabulary: DurationVocabulary,
    token_vocabulary: TokenVocabulary,
) -> None:
    sample = EncodedExercise(
        token_ids=token_vocabulary.encode([HandToken(hand=Hand.LEFT)]),
        bar_positions=[0],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("sample.mxl"),
            difficulty_level=None,
        ),
    )

    prompt = prompt_from_encoded_sample(
        sample,
        token_vocabulary=token_vocabulary,
        duration_vocabulary=duration_vocabulary,
    )

    assert prompt.tokens == [HandToken(hand=Hand.LEFT)]
    assert prompt.token_ids == sample.token_ids


def test_segment_decode_error_reports_invalid_generated_token_stream(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[HandToken(hand=Hand.RIGHT), HoldToken(duration_id=quarter_id)],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("generated"),
            difficulty_level=None,
        ),
    )

    assert segment_decode_error(segment, duration_vocabulary=duration_vocabulary) == (
        "hold token needs a previous right hand note or chord"
    )


def test_sampling_result_to_segment_counts_partial_display_bar(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    result = SamplingResult(
        tokens=[HandToken(hand=Hand.RIGHT), NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id)],
        token_ids=[],
        new_token_ids=[],
        trace=[],
        stop_reason="max_tokens",
        reached_end=False,
        constraint_error=None,
    )

    segment = sampling_result_to_segment(
        result,
        scale_root=0,
        scale_type=ScaleType.MAJOR,
        time_numerator=4,
        time_denominator=4,
    )

    assert segment.bar_count == 1
    assert segment_event_count(segment, duration_vocabulary=duration_vocabulary) == 1


def test_generation_summary_metric_rows_uses_shared_generation_metrics(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
        ],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("generated"),
            difficulty_level=None,
        ),
    )

    rows = generation_summary_metric_rows(segment, duration_vocabulary=duration_vocabulary)

    assert {"metric": "empty score", "value": False} in rows
    assert {"metric": "one hand only", "value": True} in rows
    assert {"metric": "in-scale note share", "value": "100.0%"} in rows
    assert {"metric": "hand activity balance", "value": "0.000"} in rows
    assert {"metric": "shortest note duration", "value": "1.00 beats"} in rows


def test_figure_pattern_metric_rows_summarizes_generated_figures(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=quarter_id),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=1, accidental=1, octave_offset=0, duration_id=quarter_id),
        ],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("generated"),
            difficulty_level=None,
        ),
    )

    rows = figure_pattern_metric_rows(segment, duration_vocabulary=duration_vocabulary)

    assert _row_value(rows, "single-onset figures") == 3
    assert _row_description(rows, "single-onset figures") == "Total note or chord onsets counted across both hands."
    assert _row_value(rows, "two-onset figures") == 1
    assert _row_value(rows, "unique single-onset figures") == 2
    assert _row_value(rows, "in-scale figure share") == "66.7%"
    assert _row_value(rows, "right-hand figure share") == "66.7%"


def test_figure_reference_alignment_metric_rows_compare_alignment_and_novelty(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=quarter_id),
            HandToken(hand=Hand.LEFT),
            NoteToken(degree=1, accidental=1, octave_offset=0, duration_id=quarter_id),
            NoteToken(degree=2, accidental=1, octave_offset=0, duration_id=quarter_id),
        ],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("generated"),
            difficulty_level=None,
        ),
    )
    reference_counts = {
        ScaleType.MAJOR: {
            Hand.RIGHT: {2: Counter({FigureNGram(onsets=((((0, 0),), Fraction(1)), (((1, 0),), Fraction(1)))): 4})},
            Hand.LEFT: {2: Counter({FigureNGram(onsets=((((0, 0),), Fraction(1)), (((1, 0),), Fraction(1)))): 4})},
        }
    }

    rows = figure_reference_alignment_metric_rows(
        segment,
        duration_vocabulary=duration_vocabulary,
        reference_counts=reference_counts,
        analysis_config=_analysis_config(),
    )

    assert _row_value(rows, "reference groups compared") == 2
    assert _row_value(rows, "common figure mass") == "50.0%"
    assert _row_value(rows, "novel figure mass") == "50.0%"
    assert _row_description(rows, "rhythm-shape distance") == (
        "How different the relative duration-pattern distribution is from the reference."
    )


def test_rhythm_grid_metric_rows_describe_grid_alignment(
    duration_vocabulary: DurationVocabulary,
) -> None:
    eighth_id = duration_vocabulary.fraction_to_id(Fraction(1, 8))
    segment = Segment(
        tokens=[
            HandToken(hand=Hand.RIGHT),
            NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=eighth_id),
            NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=eighth_id),
        ],
        metadata=SegmentMetadata(
            scale_root=0,
            scale_type=ScaleType.MAJOR,
            time_numerator=4,
            time_denominator=4,
            bar_count=1,
            window_start_bar=0,
            source_file=Path("generated"),
            difficulty_level=None,
        ),
    )

    rows = rhythm_grid_metric_rows(
        segment,
        duration_vocabulary=duration_vocabulary,
        analysis_config=_analysis_config(),
    )

    assert _row_value(rows, "rhythmic onsets") == 2
    assert _row_value(rows, "onset grid fit (1/16)") == "100.0%"
    assert _row_value(rows, "duration grid fit (1/16)") == "100.0%"
    assert _row_value(rows, "strong-beat onset share") == "50.0%"


def _analysis_config() -> NGramAnalysisConfig:
    return NGramAnalysisConfig(
        min_n=2,
        max_n=4,
        limit_per_group=None,
        workers=1,
        batch_size=1,
        figure_common_mass_threshold=0.8,
        rhythm_min_n=2,
        rhythm_max_n=4,
        grid_alignment_denominators=(1, 2, 4, 8, 16),
        strong_beat_offsets=(Fraction(0),),
    )


def _row_value(rows: list[dict[str, object]], metric: str) -> object:
    return _row(rows, metric)["value"]


def _row_description(rows: list[dict[str, object]], metric: str) -> object:
    return _row(rows, metric)["description"]


def _row(rows: list[dict[str, object]], metric: str) -> dict[str, object]:
    return next(row for row in rows if row["metric"] == metric)
