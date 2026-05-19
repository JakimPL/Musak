import torch
from pytest import approx

from musak_model.tokens.schema import BarToken, Hand, HandToken, NoteToken, RestToken
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.metrics import (
    BatchMetrics,
    MetricsAccumulator,
    batch_metrics_from_logits,
    build_token_kind_ids,
)


def test_batch_metrics_calculates_token_and_kind_accuracy(token_vocabulary: TokenVocabulary) -> None:
    note_a = token_vocabulary.token_to_id(NoteToken(degree=1, accidental=0, octave_offset=0, duration_id=0))
    note_b = token_vocabulary.token_to_id(NoteToken(degree=2, accidental=0, octave_offset=0, duration_id=0))
    rest = token_vocabulary.token_to_id(RestToken(duration_id=0))
    right = token_vocabulary.token_to_id(HandToken(hand=Hand.RIGHT))
    bar = token_vocabulary.token_to_id(BarToken())
    target_token_ids = torch.tensor([[note_a, rest, right, bar]])
    predicted_token_ids = torch.tensor([[note_b, rest, right, rest]])
    logits = torch.full((1, 4, token_vocabulary.vocabulary_size), -100.0)
    logits.scatter_(2, predicted_token_ids.unsqueeze(-1), 100.0)

    metrics = batch_metrics_from_logits(
        logits,
        target_token_ids=target_token_ids,
        token_padding_mask=torch.tensor([[False, False, False, False]]),
        loss=torch.tensor(1.0),
        token_kind_ids=build_token_kind_ids(token_vocabulary),
    )

    assert metrics.token_count == 4
    assert metrics.exact_match_count == 2
    assert metrics.token_kind_match_count == 3


def test_metrics_accumulator_averages_loss_accuracy_and_gradient_norms() -> None:
    accumulator = MetricsAccumulator()

    accumulator.add(
        BatchMetrics(
            loss=1.0,
            token_count=2,
            exact_match_count=1,
            token_kind_match_count=2,
            validity_penalty_loss=0.2,
            invalid_probability_mass=0.4,
            invalid_target_count=1,
            validity_penalty_token_count=2,
            cnn_gradient_norm=0.5,
            gru_gradient_norm=1.0,
            transformer_gradient_norm=1.5,
        )
    )
    accumulator.add(
        BatchMetrics(
            loss=2.0,
            token_count=6,
            exact_match_count=3,
            token_kind_match_count=4,
            validity_penalty_loss=0.6,
            invalid_probability_mass=0.8,
            invalid_target_count=2,
            validity_penalty_token_count=6,
            cnn_gradient_norm=1.5,
            gru_gradient_norm=2.0,
            transformer_gradient_norm=2.5,
        )
    )

    metrics = accumulator.to_epoch_split_metrics()

    assert metrics.loss == 1.75
    assert metrics.token_accuracy == 0.5
    assert metrics.token_kind_accuracy == 0.75
    assert metrics.validity_penalty_loss == approx(0.5)
    assert metrics.invalid_probability_mass == approx(0.7)
    assert metrics.invalid_target_rate == 0.375
    assert metrics.cnn_gradient_norm == 1.25
    assert metrics.gru_gradient_norm == 1.75
    assert metrics.transformer_gradient_norm == 2.25
