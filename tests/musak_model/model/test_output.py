import torch

from musak_model.model.output import FactorizedTokenLogits, FlatTokenAttributeBuffers, flat_token_log_scores
from musak_model.tokens.factorized import ABSENT_ATTRIBUTE_ID


def test_flat_token_log_scores_adds_only_active_attribute_heads() -> None:
    logits = FactorizedTokenLogits(
        kind=torch.tensor([[[0.0, 1.0]]]),
        degree=torch.tensor([[[2.0, 0.0]]]),
        accidental=torch.zeros(1, 1, 2),
        octave_offset=torch.zeros(1, 1, 2),
        duration=torch.tensor([[[0.5, 1.5]]]),
        hand=torch.zeros(1, 1, 2),
    )
    flat_attributes = FlatTokenAttributeBuffers(
        kind_ids=torch.tensor([0, 0]),
        degree_ids=torch.tensor([ABSENT_ATTRIBUTE_ID, 1]),
        accidental_ids=torch.tensor([ABSENT_ATTRIBUTE_ID, ABSENT_ATTRIBUTE_ID]),
        octave_offset_ids=torch.tensor([ABSENT_ATTRIBUTE_ID, ABSENT_ATTRIBUTE_ID]),
        duration_ids=torch.tensor([ABSENT_ATTRIBUTE_ID, 0]),
        hand_ids=torch.tensor([ABSENT_ATTRIBUTE_ID, ABSENT_ATTRIBUTE_ID]),
    )

    scores = flat_token_log_scores(logits, flat_attributes=flat_attributes)
    expected_kind_score = torch.log_softmax(logits.kind, dim=-1)[0, 0, 0]
    expected_degree_score = torch.log_softmax(logits.degree, dim=-1)[0, 0, 1]
    expected_duration_score = torch.log_softmax(logits.duration, dim=-1)[0, 0, 0]

    assert torch.allclose(scores[0, 0, 0], expected_kind_score)
    assert torch.allclose(scores[0, 0, 1], expected_kind_score + expected_degree_score + expected_duration_score)
