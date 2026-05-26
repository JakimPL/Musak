from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from musak_model.data.schema import Segment, SegmentMetadata
from musak_model.n_grams.figure.schema import FigureNGram
from musak_model.tokens.duration import DurationVocabulary
from musak_model.tokens.schema import EndToken, Hand, HandToken, HoldToken, NoteToken, ScaleType
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.ingestion.schema import EncodedExercise
from notebooks.utils.model_output import (
    SamplingOptions,
    SamplingResult,
    empty_prompt,
    figure_output_metric_rows,
    prompt_from_encoded_sample,
    prompt_from_text,
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


def test_figure_output_metric_rows_summarizes_generated_figures(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[HandToken(hand=Hand.RIGHT), NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id)],
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

    rows = figure_output_metric_rows(segment, duration_vocabulary=duration_vocabulary)

    assert {"metric": "generated figure groups", "value": 1} in rows
    assert {"metric": "generated figure occurrences", "value": 1} in rows
    assert {"metric": "generated unique figures", "value": 1} in rows


def test_figure_output_metric_rows_compares_with_reference_counts(
    duration_vocabulary: DurationVocabulary,
) -> None:
    quarter_id = duration_vocabulary.fraction_to_id(Fraction(1, 4))
    segment = Segment(
        tokens=[HandToken(hand=Hand.RIGHT), NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=quarter_id)],
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
            Hand.RIGHT: {
                1: Counter({FigureNGram(onsets=((((0, 0),), Fraction(1)),)): 1}),
            }
        }
    }

    rows = figure_output_metric_rows(
        segment,
        duration_vocabulary=duration_vocabulary,
        reference_counts=reference_counts,
    )

    assert {"metric": "comparable groups", "value": 1} in rows
    assert {"metric": "mean identity total variation distance", "value": "0.000"} in rows
