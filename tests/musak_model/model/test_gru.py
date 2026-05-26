from dataclasses import dataclass
from typing import Final

import pytest
import torch

from musak_model.model.config import GRUConfig
from musak_model.model.gru import BarGRUEncoder

HIDDEN_SIZE: Final[int] = 32


@dataclass(frozen=True)
class GRUShapeCase:
    label: str
    batch: int
    seq_len: int
    use_lengths: bool = False


class TestBarGRUEncoder:
    @pytest.mark.parametrize(
        "case",
        [
            GRUShapeCase(label="without_lengths", batch=4, seq_len=8, use_lengths=False),
            GRUShapeCase(label="with_lengths", batch=4, seq_len=8, use_lengths=True),
        ],
        ids=lambda case: case.label,
    )
    def test_output_shape(self, case: GRUShapeCase) -> None:
        gru = BarGRUEncoder(
            GRUConfig(enabled=True, hidden_size=HIDDEN_SIZE, num_layers=1, dropout=0.0, bidirectional=False)
        )
        inputs = torch.randn(case.batch, case.seq_len, HIDDEN_SIZE)
        lengths = torch.tensor([case.seq_len] * case.batch) if case.use_lengths else None
        outputs = gru(inputs, lengths=lengths)
        assert outputs.shape == (case.batch, HIDDEN_SIZE)

    def test_packed_ignores_padding(self) -> None:
        torch.manual_seed(42)
        gru = BarGRUEncoder(
            GRUConfig(enabled=True, hidden_size=HIDDEN_SIZE, num_layers=1, dropout=0.0, bidirectional=False)
        )
        gru.eval()

        real_tokens = torch.randn(1, 3, HIDDEN_SIZE)
        padded = torch.cat([real_tokens, torch.zeros(1, 5, HIDDEN_SIZE)], dim=1)

        with torch.no_grad():
            real_outputs = gru(real_tokens, lengths=torch.tensor([3]))
            padded_outputs = gru(padded, lengths=torch.tensor([3]))

        assert torch.allclose(real_outputs, padded_outputs, atol=1e-6)
