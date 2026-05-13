from dataclasses import dataclass
from typing import Final

import pytest
import torch
from torch import Tensor

from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import CNNConfig, ConditioningConfig, GRUConfig, ModelConfig, TransformerConfig
from musak_model.model.gru import BarGRUEncoder

VOCAB: Final[int] = 64
H: Final[int] = 32  # hidden size (small for speed)


def _small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB,
        cnn=CNNConfig(out_channels=H, kernel_sizes=(3,), num_layers=1, dropout=0.0),
        gru=GRUConfig(hidden_size=H, num_layers=1, dropout=0.0, bidirectional=False),
        transformer=TransformerConfig(
            hidden_size=H,
            num_heads=2,
            num_layers=1,
            feedforward_size=64,
            dropout=0.0,
            max_sequence_length=128,
        ),
        conditioning=ConditioningConfig(
            num_difficulty_levels=6,
            num_scale_types=2,
            num_time_signatures=5,
            cfg_dropout_probability=0.0,
        ),
    )


def _uniform_bar_positions(batch: int, seq_len: int, num_bars: int) -> Tensor:
    tokens_per_bar = seq_len // num_bars
    return torch.arange(seq_len).div(tokens_per_bar, rounding_mode="floor").clamp(max=num_bars - 1).expand(batch, -1)


@dataclass(frozen=True)
class ForwardShapeCase:
    label: str
    batch: int
    seq_len: int
    num_bars: int
    with_all_conditioning: bool = False


@dataclass(frozen=True)
class BarLayoutCase:
    label: str
    token_shape: tuple[int, int]
    bar_positions_rows: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ValidationCase:
    label: str
    match: str
    token_seq_len: int = 16
    bar_seq_len: int = 16  # different from token_seq_len triggers shape-mismatch error
    bar_ndim: int = 2  # 1 triggers ndim error
    difficulty_values: tuple[int, ...] | None = None
    difficulty_ndim: int = 1  # 2 triggers 2D-tensor error


@dataclass(frozen=True)
class GRUShapeCase:
    label: str
    batch: int
    seq_len: int
    use_lengths: bool = False


class TestForwardOutputShape:
    @pytest.mark.parametrize(
        "case",
        [
            ForwardShapeCase(label="standard", batch=2, seq_len=32, num_bars=4),
            ForwardShapeCase(label="batch_size_1", batch=1, seq_len=16, num_bars=2),
            ForwardShapeCase(label="all_conditioning", batch=3, seq_len=24, num_bars=3, with_all_conditioning=True),
        ],
        ids=lambda c: c.label,
    )
    def test_output_shape(self, case: ForwardShapeCase) -> None:
        config = _small_config()
        model = HierarchicalAutoregressiveModel(config)
        token_ids = torch.randint(0, VOCAB, (case.batch, case.seq_len))
        bar_positions = _uniform_bar_positions(case.batch, case.seq_len, case.num_bars)

        kwargs: dict[str, Tensor] = {}
        if case.with_all_conditioning:
            kwargs["difficulty_ids"] = torch.zeros(case.batch, dtype=torch.long)
            kwargs["scale_type_ids"] = torch.full(
                (case.batch,), config.conditioning.num_scale_types - 1, dtype=torch.long
            )
            kwargs["time_signature_ids"] = torch.full(
                (case.batch,), config.conditioning.num_time_signatures - 1, dtype=torch.long
            )

        logits = model(token_ids, bar_positions=bar_positions, **kwargs)
        assert logits.shape == (case.batch, case.seq_len, VOCAB)


class TestForwardBarLayouts:
    @pytest.mark.parametrize(
        "case",
        [
            BarLayoutCase(
                label="uneven_bars_4_vs_2",
                token_shape=(2, 32),
                bar_positions_rows=(
                    (0,) * 8 + (1,) * 8 + (2,) * 8 + (3,) * 8,
                    (0,) * 16 + (1,) * 16,
                ),
            ),
            BarLayoutCase(
                label="single_bar_all_tokens",
                token_shape=(2, 16),
                bar_positions_rows=(
                    (0,) * 16,
                    (0,) * 16,
                ),
            ),
        ],
        ids=lambda c: c.label,
    )
    def test_output_shape(self, case: BarLayoutCase) -> None:
        model = HierarchicalAutoregressiveModel(_small_config())
        token_ids = torch.randint(0, VOCAB, case.token_shape)
        bar_positions = torch.tensor(list(case.bar_positions_rows))
        logits = model(token_ids, bar_positions=bar_positions)
        assert logits.shape == (*case.token_shape, VOCAB)


class TestForwardValidation:
    _BATCH_SIZE: Final[int] = 2

    def _build_token_ids(self, case: ValidationCase) -> Tensor:
        return torch.randint(0, VOCAB, (self._BATCH_SIZE, case.token_seq_len))

    def _build_bar_positions(self, case: ValidationCase) -> Tensor:
        if case.bar_ndim == 1:
            return torch.zeros(self._BATCH_SIZE * case.bar_seq_len, dtype=torch.long)
        return torch.zeros(self._BATCH_SIZE, case.bar_seq_len, dtype=torch.long)

    def _build_forward_kwargs(self, case: ValidationCase) -> dict[str, Tensor]:
        kwargs: dict[str, Tensor] = {}
        if case.difficulty_values is None:
            return kwargs

        difficulty_ids = torch.tensor(case.difficulty_values)
        if case.difficulty_ndim == 2:
            difficulty_ids = difficulty_ids.unsqueeze(1)

        kwargs["difficulty_ids"] = difficulty_ids
        return kwargs

    @pytest.mark.parametrize(
        "case",
        [
            ValidationCase(
                label="bar_positions_shape_mismatch",
                match="bar_positions shape",
                bar_seq_len=12,  # mismatches token_seq_len=16
            ),
            ValidationCase(
                label="bar_positions_wrong_ndim",
                match="2 dimensions",
                bar_ndim=1,
            ),
            ValidationCase(
                label="conditioning_wrong_batch_size",
                match="batch size",
                difficulty_values=(0, 0, 0),  # 3 values for batch of 2
            ),
            ValidationCase(
                label="conditioning_negative_value",
                match="negative",
                difficulty_values=(-1, 0),
            ),
            ValidationCase(
                label="conditioning_out_of_range_value",
                match="outside range",
                difficulty_values=(0, 6),  # 6 == num_difficulty_levels default, out of range
            ),
            ValidationCase(
                label="conditioning_2d_tensor",
                match="1D tensor",
                difficulty_values=(0, 0),
                difficulty_ndim=2,
            ),
        ],
        ids=lambda c: c.label,
    )
    def test_raises_value_error(self, case: ValidationCase) -> None:
        """Assert that each malformed validation case fails with a ValueError.

        The test follows arrange/act/assert explicitly via helper methods:
        1) build model and synthetic inputs for this case,
        2) execute forward with potentially invalid arguments,
        3) assert the expected validation error message fragment.
        """
        model = HierarchicalAutoregressiveModel(_small_config())
        token_ids = self._build_token_ids(case)
        bar_positions = self._build_bar_positions(case)
        kwargs = self._build_forward_kwargs(case)

        with pytest.raises(ValueError, match=case.match):
            model(token_ids, bar_positions=bar_positions, **kwargs)


class TestForwardBehaviour:
    def test_is_deterministic(self) -> None:
        torch.manual_seed(0)
        model = HierarchicalAutoregressiveModel(_small_config())
        model.eval()
        token_ids = torch.randint(0, VOCAB, (2, 16))
        bar_positions = _uniform_bar_positions(2, 16, 2)
        with torch.no_grad():
            out1 = model(token_ids, bar_positions=bar_positions)
            out2 = model(token_ids, bar_positions=bar_positions)

        assert torch.equal(out1, out2)

    def test_gradients_flow_through_all_parameters(self) -> None:
        """Verify that a backward pass reaches every trainable parameter.

        The scalar loss is built as ``logits.sum()`` so autograd can propagate
        through the full forward graph without needing task labels. After
        ``backward()``, every parameter is expected to have a non-None gradient;
        missing gradients usually indicate a disconnected branch in the model.
        """
        model = HierarchicalAutoregressiveModel(_small_config())
        token_ids = torch.randint(0, VOCAB, (2, 16))
        bar_positions = _uniform_bar_positions(2, 16, 2)
        logits = model(token_ids, bar_positions=bar_positions)
        logits.sum().backward()
        params_without_grad = [name for name, p in model.named_parameters() if p.grad is None]
        assert params_without_grad == [], f"No gradient for: {params_without_grad}"


class TestBarGRUEncoder:
    @pytest.mark.parametrize(
        "case",
        [
            GRUShapeCase(label="without_lengths", batch=4, seq_len=8, use_lengths=False),
            GRUShapeCase(label="with_lengths", batch=4, seq_len=8, use_lengths=True),
        ],
        ids=lambda c: c.label,
    )
    def test_output_shape(self, case: GRUShapeCase) -> None:
        gru = BarGRUEncoder(GRUConfig(hidden_size=H, num_layers=1, dropout=0.0, bidirectional=False))
        x = torch.randn(case.batch, case.seq_len, H)
        lengths = torch.tensor([case.seq_len] * case.batch) if case.use_lengths else None
        out = gru(x, lengths=lengths)
        assert out.shape == (case.batch, H)

    def test_packed_ignores_padding(self) -> None:
        torch.manual_seed(42)
        gru = BarGRUEncoder(GRUConfig(hidden_size=H, num_layers=1, dropout=0.0, bidirectional=False))
        gru.eval()

        real_tokens = torch.randn(1, 3, H)
        padded = torch.cat([real_tokens, torch.zeros(1, 5, H)], dim=1)  # same content, trailing zeros

        with torch.no_grad():
            out_real = gru(real_tokens, lengths=torch.tensor([3]))
            out_padded = gru(padded, lengths=torch.tensor([3]))

        assert torch.allclose(out_real, out_padded, atol=1e-6)
