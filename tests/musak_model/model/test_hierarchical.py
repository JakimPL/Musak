from dataclasses import dataclass
from typing import Final

import pytest
import torch
from torch import Tensor

from musak_model.auxiliary.config import MusicalAuxiliaryTargetConfig
from musak_model.auxiliary.schema import MusicalAuxiliaryLogits
from musak_model.conditioning.config import ConditioningConfig, DifficultyConfig, HarmonicConditioningConfig
from musak_model.conditioning.harmony.fields import HARMONIC_PLAN_TENSOR_FIELDS
from musak_model.conditioning.harmony.schema import HarmonicPlanInputTensors
from musak_model.conditioning.time_signature import TimeSignatureVocabularyConfig
from musak_model.model import HierarchicalAutoregressiveModel
from musak_model.model.config import (
    CNNConfig,
    GRUConfig,
    ModelConfig,
    ModelInputConfig,
    ModelOutputConfig,
    ModelOutputMode,
    TokenInputEmbeddingMode,
    TransformerConfig,
)
from musak_model.tokens.factorized import TOKEN_KIND_COUNT, flat_vocabulary_attributes

VOCAB: Final[int] = 64
H: Final[int] = 32  # hidden size (small for speed)
FACTORIZED_DURATION_VOCAB: Final[int] = 1
FACTORIZED_VOCAB: Final[int] = len(flat_vocabulary_attributes(duration_vocabulary_size=FACTORIZED_DURATION_VOCAB))


def _musical_auxiliary_target_config() -> MusicalAuxiliaryTargetConfig:
    return MusicalAuxiliaryTargetConfig(
        note_density_bucket_boundaries=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
        rhythmic_diversity_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
        voice_independence_bucket_boundaries=(0.2, 0.4, 0.6, 0.8),
        hand_span_bucket_boundaries=(3, 5, 8, 12, 16),
    )


def _small_config(
    *,
    output_mode: ModelOutputMode = ModelOutputMode.FLAT,
    input_embedding_mode: TokenInputEmbeddingMode = TokenInputEmbeddingMode.FLAT,
    vocabulary_size: int = VOCAB,
    duration_vocabulary_size: int = 1,
    harmony_enabled: bool = False,
) -> ModelConfig:
    return ModelConfig(
        vocabulary_size=vocabulary_size,
        duration_vocabulary_size=duration_vocabulary_size,
        input=ModelInputConfig(embedding_mode=input_embedding_mode),
        output=ModelOutputConfig(mode=output_mode),
        musical_auxiliary_targets=_musical_auxiliary_target_config(),
        cnn=CNNConfig(enabled=True, out_channels=H, kernel_sizes=(3,), num_layers=1, dropout=0.0),
        gru=GRUConfig(enabled=True, hidden_size=H, num_layers=1, dropout=0.0, bidirectional=False),
        transformer=TransformerConfig(
            hidden_size=H,
            num_heads=2,
            num_layers=1,
            feedforward_size=64,
            dropout=0.0,
            max_sequence_length=128,
        ),
        conditioning=ConditioningConfig(
            difficulty=DifficultyConfig(max_level=5),
            time_signature=TimeSignatureVocabularyConfig(max_denominator=4, relative_numerator_range=2),
            harmony=HarmonicConditioningConfig(enabled=harmony_enabled),
            cfg_dropout_probability=0.0,
        ),
    )


def _small_config_with_encoders(*, cnn_enabled: bool, gru_enabled: bool) -> ModelConfig:
    config = _small_config()
    return config.model_copy(
        update={
            "cnn": config.cnn.model_copy(update={"enabled": cnn_enabled}),
            "gru": config.gru.model_copy(update={"enabled": gru_enabled}),
        }
    )


def _uniform_bar_positions(batch: int, seq_len: int, num_bars: int) -> Tensor:
    tokens_per_bar = seq_len // num_bars
    return torch.arange(seq_len).div(tokens_per_bar, rounding_mode="floor").clamp(max=num_bars - 1).expand(batch, -1)


def _coordinate_kwargs(token_ids: Tensor) -> dict[str, Tensor]:
    return {
        "bar_relative_ticks": torch.zeros_like(token_ids),
        "bar_duration_ticks": torch.ones_like(token_ids),
        "active_hand_ids": torch.zeros_like(token_ids),
    }


def _training_logits_kwargs(token_ids: Tensor, *, bar_positions: Tensor) -> dict[str, Tensor]:
    return {
        **_coordinate_kwargs(token_ids),
        "target_bar_positions": bar_positions,
        "bar_counts": bar_positions.max(dim=1).values + 1,
    }


def _harmonic_plan(token_ids: Tensor) -> HarmonicPlanInputTensors:
    ids = torch.ones_like(token_ids)
    return HarmonicPlanInputTensors(**{field.name: ids for field in HARMONIC_PLAN_TENSOR_FIELDS})


def _musical_auxiliary_logits_sum(logits: MusicalAuxiliaryLogits) -> Tensor:
    return (
        logits.note_density.sum()
        + logits.rhythmic_diversity.sum()
        + logits.voice_independence.sum()
        + logits.uses_accidentals.sum()
        + logits.dotted_duration.sum()
        + logits.hand_span.sum()
        + logits.bar.note_density.sum()
        + logits.bar.rhythmic_diversity.sum()
        + logits.bar.voice_independence.sum()
        + logits.bar.uses_accidentals.sum()
        + logits.bar.dotted_duration.sum()
        + logits.bar.hand_span.sum()
    )


def _backward(loss: Tensor) -> None:
    loss.backward()  # type: ignore[no-untyped-call]


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
class EncoderBypassCase:
    label: str
    cnn_enabled: bool
    gru_enabled: bool
    omitted_parameter_prefixes: tuple[str, ...]


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

        logits = model(token_ids, bar_positions=bar_positions, **_coordinate_kwargs(token_ids), **kwargs)
        assert logits.shape == (case.batch, case.seq_len, VOCAB)

    def test_factorized_output_mode_returns_flat_token_scores_and_factorized_heads(self) -> None:
        config = _small_config(
            output_mode=ModelOutputMode.FACTORIZED,
            vocabulary_size=FACTORIZED_VOCAB,
            duration_vocabulary_size=FACTORIZED_DURATION_VOCAB,
        )
        model = HierarchicalAutoregressiveModel(config)
        token_ids = torch.randint(0, FACTORIZED_VOCAB, (2, 8))
        bar_positions = _uniform_bar_positions(2, 8, 2)

        flat_scores = model(token_ids, bar_positions=bar_positions, **_coordinate_kwargs(token_ids))
        factorized_logits = model.factorized_logits(
            token_ids,
            bar_positions=bar_positions,
            **_coordinate_kwargs(token_ids),
        )
        training_logits = model.training_logits(
            token_ids,
            bar_positions=bar_positions,
            **_training_logits_kwargs(token_ids, bar_positions=bar_positions),
        )

        assert flat_scores.shape == (2, 8, FACTORIZED_VOCAB)
        assert training_logits.flat_logits.shape == (2, 8, FACTORIZED_VOCAB)
        assert training_logits.factorized_logits is not None
        assert factorized_logits.kind.shape == (2, 8, TOKEN_KIND_COUNT)
        assert factorized_logits.duration.shape == (2, 8, FACTORIZED_DURATION_VOCAB)
        assert training_logits.musical_auxiliary_logits.bar.note_density.shape == (2, 2, 7)

    def test_factorized_input_embedding_mode_returns_flat_output_shape(self) -> None:
        config = _small_config(
            input_embedding_mode=TokenInputEmbeddingMode.FLAT_PLUS_FACTORIZED,
            vocabulary_size=FACTORIZED_VOCAB,
            duration_vocabulary_size=FACTORIZED_DURATION_VOCAB,
        )
        model = HierarchicalAutoregressiveModel(config)
        token_ids = torch.randint(0, FACTORIZED_VOCAB, (2, 8))
        bar_positions = _uniform_bar_positions(2, 8, 2)

        logits = model(token_ids, bar_positions=bar_positions, **_coordinate_kwargs(token_ids))

        assert logits.shape == (2, 8, FACTORIZED_VOCAB)

    def test_factorized_input_embedding_mode_requires_matching_vocabulary_shape(self) -> None:
        config = _small_config(input_embedding_mode=TokenInputEmbeddingMode.FLAT_PLUS_FACTORIZED)

        with pytest.raises(ValueError, match="factorized input attribute table size"):
            HierarchicalAutoregressiveModel(config)

    def test_harmonic_plan_conditioning_returns_flat_output_shape(self) -> None:
        config = _small_config(harmony_enabled=True)
        model = HierarchicalAutoregressiveModel(config)
        token_ids = torch.randint(0, VOCAB, (2, 8))
        bar_positions = _uniform_bar_positions(2, 8, 2)

        logits = model(
            token_ids,
            bar_positions=bar_positions,
            **_coordinate_kwargs(token_ids),
            harmonic_plan=_harmonic_plan(token_ids),
        )

        assert logits.shape == (2, 8, VOCAB)

    def test_harmonic_plan_inputs_require_enabled_conditioning(self) -> None:
        config = _small_config(harmony_enabled=False)
        model = HierarchicalAutoregressiveModel(config)
        token_ids = torch.randint(0, VOCAB, (2, 8))
        bar_positions = _uniform_bar_positions(2, 8, 2)

        with pytest.raises(ValueError, match="harmony conditioning"):
            model(
                token_ids,
                bar_positions=bar_positions,
                **_coordinate_kwargs(token_ids),
                harmonic_plan=_harmonic_plan(token_ids),
            )


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
        config = _small_config()
        model = HierarchicalAutoregressiveModel(config)
        token_ids = torch.randint(0, VOCAB, case.token_shape)
        bar_positions = torch.tensor(list(case.bar_positions_rows))
        logits = model(token_ids, bar_positions=bar_positions, **_coordinate_kwargs(token_ids))
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
            ValidationCase(
                label="sequence_exceeds_transformer_max_length",
                match="max_sequence_length",
                token_seq_len=129,
                bar_seq_len=129,
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
        config = _small_config()
        model = HierarchicalAutoregressiveModel(config)
        token_ids = self._build_token_ids(case)
        bar_positions = self._build_bar_positions(case)
        kwargs = self._build_forward_kwargs(case)

        with pytest.raises(ValueError, match=case.match):
            model(token_ids, bar_positions=bar_positions, **_coordinate_kwargs(token_ids), **kwargs)


class TestForwardBehaviour:
    def test_is_deterministic(self) -> None:
        torch.manual_seed(0)
        config = _small_config()
        model = HierarchicalAutoregressiveModel(config)
        model.eval()
        token_ids = torch.randint(0, VOCAB, (2, 16))
        bar_positions = _uniform_bar_positions(2, 16, 2)
        with torch.no_grad():
            out1 = model(token_ids, bar_positions=bar_positions, **_coordinate_kwargs(token_ids))
            out2 = model(token_ids, bar_positions=bar_positions, **_coordinate_kwargs(token_ids))

        assert torch.equal(out1, out2)

    def test_gradients_flow_through_all_parameters(self) -> None:
        """Verify that a backward pass reaches every trainable parameter.

        The scalar loss is built as ``logits.sum()`` so autograd can propagate
        through the full forward graph without needing task labels. After
        ``backward()``, every parameter is expected to have a non-None gradient;
        missing gradients usually indicate a disconnected branch in the model.
        """
        config = _small_config()
        model = HierarchicalAutoregressiveModel(config)
        token_ids = torch.randint(0, VOCAB, (2, 16))
        bar_positions = _uniform_bar_positions(2, 16, 2)
        logits = model.training_logits(
            token_ids,
            bar_positions=bar_positions,
            **_training_logits_kwargs(token_ids, bar_positions=bar_positions),
            difficulty_ids=torch.zeros(2, dtype=torch.long),
            scale_type_ids=torch.zeros(2, dtype=torch.long),
            time_signature_ids=torch.zeros(2, dtype=torch.long),
            structural_control_ids=torch.zeros(
                2,
                len(config.conditioning.structural_vocabulary_sizes),
                dtype=torch.long,
            ),
        )
        _backward(logits.flat_logits.sum() + _musical_auxiliary_logits_sum(logits.musical_auxiliary_logits))
        params_without_grad = [name for name, p in model.named_parameters() if p.grad is None]
        assert params_without_grad == [], f"No gradient for: {params_without_grad}"

    def test_gradients_flow_through_factorized_input_embeddings(self) -> None:
        config = _small_config(
            input_embedding_mode=TokenInputEmbeddingMode.FLAT_PLUS_FACTORIZED,
            vocabulary_size=FACTORIZED_VOCAB,
            duration_vocabulary_size=FACTORIZED_DURATION_VOCAB,
        )
        model = HierarchicalAutoregressiveModel(config)
        token_ids = torch.arange(16, dtype=torch.long).remainder(FACTORIZED_VOCAB).view(2, 8)
        bar_positions = _uniform_bar_positions(2, 8, 2)
        logits = model.training_logits(
            token_ids,
            bar_positions=bar_positions,
            **_training_logits_kwargs(token_ids, bar_positions=bar_positions),
        )

        _backward(logits.flat_logits.sum() + _musical_auxiliary_logits_sum(logits.musical_auxiliary_logits))

        gradient_names = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
        assert "_input_embeddings._kind_embedding.weight" in gradient_names
        assert "_input_embeddings._duration_embedding.weight" in gradient_names
        assert "_input_embeddings._degree_embedding.weight" in gradient_names

    def test_gradients_flow_through_harmonic_plan_embeddings(self) -> None:
        config = _small_config(harmony_enabled=True)
        model = HierarchicalAutoregressiveModel(config)
        token_ids = torch.randint(0, VOCAB, (2, 8))
        bar_positions = _uniform_bar_positions(2, 8, 2)
        logits = model.training_logits(
            token_ids,
            bar_positions=bar_positions,
            **_training_logits_kwargs(token_ids, bar_positions=bar_positions),
            harmonic_plan=_harmonic_plan(token_ids),
        )

        _backward(logits.flat_logits.sum() + _musical_auxiliary_logits_sum(logits.musical_auxiliary_logits))

        gradient_names = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
        for field in HARMONIC_PLAN_TENSOR_FIELDS:
            assert f"_harmonic_plan_embeddings._embeddings.{field.name}.weight" in gradient_names

    def test_future_tokens_do_not_change_earlier_logits(self) -> None:
        torch.manual_seed(0)
        model = HierarchicalAutoregressiveModel(_small_config())
        model.eval()
        token_ids = torch.randint(0, VOCAB, (2, 16))
        changed_token_ids = token_ids.clone()
        changed_token_ids[:, 13] = (changed_token_ids[:, 13] + 1) % VOCAB
        bar_positions = _uniform_bar_positions(2, 16, 2)

        with torch.no_grad():
            original_logits = model(token_ids, bar_positions=bar_positions, **_coordinate_kwargs(token_ids))
            changed_logits = model(
                changed_token_ids,
                bar_positions=bar_positions,
                **_coordinate_kwargs(changed_token_ids),
            )

        assert torch.allclose(original_logits[:, :13], changed_logits[:, :13], atol=1e-6)


class TestEncoderBypass:
    @pytest.mark.parametrize(
        "case",
        [
            EncoderBypassCase(
                label="cnn_disabled",
                cnn_enabled=False,
                gru_enabled=True,
                omitted_parameter_prefixes=("_to_local_hidden", "_local_encoder"),
            ),
            EncoderBypassCase(
                label="gru_disabled",
                cnn_enabled=True,
                gru_enabled=False,
                omitted_parameter_prefixes=(
                    "_to_bar_hidden",
                    "_bar_prefix_encoder",
                    "_bar_encoder",
                    "_bar_prefix_to_transformer_hidden",
                    "_bar_to_transformer_hidden",
                ),
            ),
            EncoderBypassCase(
                label="cnn_and_gru_disabled",
                cnn_enabled=False,
                gru_enabled=False,
                omitted_parameter_prefixes=(
                    "_to_local_hidden",
                    "_local_encoder",
                    "_to_bar_hidden",
                    "_bar_prefix_encoder",
                    "_bar_encoder",
                    "_bar_prefix_to_transformer_hidden",
                    "_bar_to_transformer_hidden",
                    "_local_to_transformer_hidden",
                ),
            ),
        ],
        ids=lambda case: case.label,
    )
    def test_omits_disabled_encoder_parameters(self, case: EncoderBypassCase) -> None:
        config = _small_config_with_encoders(cnn_enabled=case.cnn_enabled, gru_enabled=case.gru_enabled)
        model = HierarchicalAutoregressiveModel(config)

        parameter_names = tuple(name for name, _parameter in model.named_parameters())

        for prefix in case.omitted_parameter_prefixes:
            assert not any(name.startswith(prefix) for name in parameter_names)

    @pytest.mark.parametrize(
        "case",
        [
            EncoderBypassCase(
                label="cnn_disabled",
                cnn_enabled=False,
                gru_enabled=True,
                omitted_parameter_prefixes=(),
            ),
            EncoderBypassCase(
                label="gru_disabled",
                cnn_enabled=True,
                gru_enabled=False,
                omitted_parameter_prefixes=(),
            ),
            EncoderBypassCase(
                label="cnn_and_gru_disabled",
                cnn_enabled=False,
                gru_enabled=False,
                omitted_parameter_prefixes=(),
            ),
        ],
        ids=lambda case: case.label,
    )
    def test_forward_and_gradients_work(self, case: EncoderBypassCase) -> None:
        config = _small_config_with_encoders(cnn_enabled=case.cnn_enabled, gru_enabled=case.gru_enabled)
        model = HierarchicalAutoregressiveModel(config)
        token_ids = torch.randint(0, VOCAB, (2, 16))
        bar_positions = _uniform_bar_positions(2, 16, 2)

        logits = model.training_logits(
            token_ids,
            bar_positions=bar_positions,
            **_training_logits_kwargs(token_ids, bar_positions=bar_positions),
            difficulty_ids=torch.zeros(2, dtype=torch.long),
            scale_type_ids=torch.zeros(2, dtype=torch.long),
            time_signature_ids=torch.zeros(2, dtype=torch.long),
            structural_control_ids=torch.zeros(
                2,
                len(config.conditioning.structural_vocabulary_sizes),
                dtype=torch.long,
            ),
        )
        _backward(logits.flat_logits.sum() + _musical_auxiliary_logits_sum(logits.musical_auxiliary_logits))

        params_without_grad = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        assert logits.flat_logits.shape == (2, 16, VOCAB)
        assert params_without_grad == [], f"No gradient for: {params_without_grad}"
