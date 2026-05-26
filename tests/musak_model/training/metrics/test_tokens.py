import torch

from musak_model.tokens.schema import BarToken, Hand, HandToken, NoteToken, RestToken
from musak_model.tokens.vocabulary import TokenVocabulary
from musak_model.training.metrics import batch_metrics_from_logits, build_token_kind_ids


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
