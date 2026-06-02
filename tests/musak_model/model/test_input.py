import torch

from musak_model.model.input import _optional_embedding_ids
from musak_model.tokens.factorized import ABSENT_ATTRIBUTE_ID


def test_optional_embedding_ids_maps_absent_attributes_to_padding_bucket() -> None:
    attribute_ids = torch.tensor([ABSENT_ATTRIBUTE_ID, 0, 2])

    embedding_ids = _optional_embedding_ids(attribute_ids)

    assert torch.equal(embedding_ids, torch.tensor([0, 1, 3]))
