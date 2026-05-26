from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from musak_model.model.cnn import LocalConvEncoder
from musak_model.model.config import ModelConfig
from musak_model.model.gru import BarGRUEncoder, BarPrefixGRUEncoder
from musak_model.model.transformer import CausalTransformerDecoder


class HierarchicalAutoregressiveModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._config = config

        self._token_embedding = nn.Embedding(config.vocabulary_size, config.transformer.hidden_size)
        local_hidden_size = config.cnn.out_channels if config.cnn.enabled else config.transformer.hidden_size
        if config.cnn.enabled:
            self._to_local_hidden = nn.Linear(config.transformer.hidden_size, config.cnn.out_channels)
            self._local_encoder = LocalConvEncoder(config.cnn)

        if config.gru.enabled:
            self._to_bar_hidden = nn.Linear(local_hidden_size, config.gru.hidden_size)
            self._bar_prefix_encoder = BarPrefixGRUEncoder(config.gru)
            self._bar_encoder = BarGRUEncoder(config.gru)
            self._bar_prefix_to_transformer_hidden = nn.Linear(config.gru.hidden_size, config.transformer.hidden_size)
            self._bar_to_transformer_hidden = nn.Linear(config.gru.hidden_size, config.transformer.hidden_size)
        elif config.cnn.enabled:
            self._local_to_transformer_hidden = nn.Linear(local_hidden_size, config.transformer.hidden_size)

        self._decoder = CausalTransformerDecoder(config.transformer)
        self._lm_head = nn.Linear(config.transformer.hidden_size, config.vocabulary_size)

        self._difficulty_embedding = nn.Embedding(
            config.conditioning.num_difficulty_levels, config.transformer.hidden_size
        )
        self._scale_type_embedding = nn.Embedding(config.conditioning.num_scale_types, config.transformer.hidden_size)
        self._time_signature_embedding = nn.Embedding(
            config.conditioning.num_time_signatures, config.transformer.hidden_size
        )
        self._structural_control_embeddings = nn.ModuleList(
            nn.Embedding(vocabulary_size, config.transformer.hidden_size)
            for vocabulary_size in config.conditioning.structural_vocabulary_sizes
        )
        self._conditioning_norm = nn.LayerNorm(config.transformer.hidden_size)

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
        token_embeddings = self._token_embedding(token_ids)
        self._validate_bar_input_shapes(bar_embeddings=token_embeddings, bar_positions=bar_positions)
        conditioning_prefix = self._build_conditioning_prefix(
            batch_size=token_ids.size(0),
            device=token_ids.device,
            difficulty_ids=difficulty_ids,
            scale_type_ids=scale_type_ids,
            time_signature_ids=time_signature_ids,
            structural_control_ids=structural_control_ids,
        )

        target_padding_mask = self._build_target_padding_mask(
            bar_positions=bar_positions,
            token_padding_mask=token_padding_mask,
        )
        local_embeddings = self._local_embeddings(token_embeddings)
        decoder_inputs, memory_context, memory_padding_mask, memory_attention_mask = self._decoder_context(
            token_embeddings=token_embeddings,
            local_embeddings=local_embeddings,
            conditioning_prefix=conditioning_prefix,
            bar_positions=bar_positions,
        )
        decoded_embeddings = self._decoder(
            decoder_inputs,
            memory_context,
            target_padding_mask=target_padding_mask,
            memory_padding_mask=memory_padding_mask,
            memory_attention_mask=memory_attention_mask,
        )
        return cast(Tensor, self._lm_head(decoded_embeddings))

    def _local_embeddings(self, token_embeddings: Tensor) -> Tensor:
        if not self._config.cnn.enabled:
            return token_embeddings

        return cast(Tensor, self._local_encoder(self._to_local_hidden(token_embeddings)))

    def _decoder_context(
        self,
        *,
        token_embeddings: Tensor,
        local_embeddings: Tensor,
        conditioning_prefix: Tensor,
        bar_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self._config.gru.enabled:
            return self._decoder_context_with_bars(
                token_embeddings=token_embeddings,
                local_embeddings=local_embeddings,
                conditioning_prefix=conditioning_prefix,
                bar_positions=bar_positions,
            )

        decoder_inputs = token_embeddings
        if self._config.cnn.enabled:
            decoder_inputs = decoder_inputs + self._local_to_transformer_hidden(local_embeddings)

        return (
            decoder_inputs,
            conditioning_prefix,
            self._build_conditioning_memory_padding_mask(conditioning_prefix),
            self._build_conditioning_memory_attention_mask(bar_positions=bar_positions),
        )

    def _decoder_context_with_bars(
        self,
        *,
        token_embeddings: Tensor,
        local_embeddings: Tensor,
        conditioning_prefix: Tensor,
        bar_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        bar_embeddings = self._to_bar_hidden(local_embeddings)
        bar_prefixes, bar_context, bar_memory_padding_mask = self._encode_bar_representations(
            bar_embeddings=bar_embeddings,
            bar_positions=bar_positions,
        )
        transformer_bar_prefixes = self._bar_prefix_to_transformer_hidden(bar_prefixes)
        transformer_bar_context = self._bar_to_transformer_hidden(bar_context)
        memory_context = torch.cat([conditioning_prefix, transformer_bar_context], dim=1)
        memory_padding_mask = self._build_memory_padding_mask(
            bar_memory_padding_mask=bar_memory_padding_mask,
        )
        memory_attention_mask = self._build_memory_attention_mask(
            bar_positions=bar_positions,
            num_bar_memory=transformer_bar_context.size(1),
        )
        return token_embeddings + transformer_bar_prefixes, memory_context, memory_padding_mask, memory_attention_mask

    def _encode_bar_representations(
        self,
        *,
        bar_embeddings: Tensor,
        bar_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._validate_bar_input_shapes(bar_embeddings=bar_embeddings, bar_positions=bar_positions)

        bar_groups, group_batch_indices, group_token_indices = self._collect_bar_groups_with_indices(
            bar_embeddings=bar_embeddings,
            bar_positions=bar_positions,
        )
        lengths = torch.tensor([group.size(0) for group in bar_groups], dtype=torch.long)
        padded_groups = pad_sequence(bar_groups, batch_first=True)

        all_prefixes = self._bar_prefix_encoder(padded_groups, lengths=lengths)
        all_bar_vectors = self._bar_encoder(padded_groups, lengths=lengths)
        bar_prefixes = self._scatter_prefix_outputs(
            all_prefixes=all_prefixes,
            group_batch_indices=group_batch_indices,
            group_token_indices=group_token_indices,
            output_shape=bar_embeddings.shape,
        )
        per_batch_vectors = self._split_vectors_per_item(
            all_bar_vectors=all_bar_vectors,
            group_batch_indices=group_batch_indices,
            batch_size=bar_embeddings.size(0),
        )
        bar_context = self._pad_and_stack_to_batch(per_batch_vectors)
        bar_memory_padding_mask = self._build_bar_memory_padding_mask(
            per_batch_vectors=per_batch_vectors,
            max_bars=bar_context.size(1),
            device=bar_embeddings.device,
        )
        return bar_prefixes, bar_context, bar_memory_padding_mask

    def _collect_bar_groups_with_indices(
        self,
        *,
        bar_embeddings: Tensor,
        bar_positions: Tensor,
    ) -> tuple[list[Tensor], Tensor, list[Tensor]]:
        batch_size = bar_embeddings.size(0)
        hidden_size = bar_embeddings.size(-1)
        groups: list[Tensor] = []
        group_batches: list[int] = []
        group_token_indices: list[Tensor] = []

        for batch_index in range(batch_size):
            positions = bar_positions[batch_index]
            valid_bar_indices = torch.unique(positions[positions >= 0], sorted=True)
            for bar_index in valid_bar_indices.tolist():
                token_indices = torch.nonzero(positions == int(bar_index), as_tuple=False).squeeze(1)
                groups.append(bar_embeddings[batch_index, token_indices])
                group_batches.append(batch_index)
                group_token_indices.append(token_indices)

        present_batches = set(group_batches)
        for batch_index in range(batch_size):
            if batch_index in present_batches:
                continue
            groups.append(torch.zeros(1, hidden_size, device=bar_embeddings.device, dtype=bar_embeddings.dtype))
            group_batches.append(batch_index)
            group_token_indices.append(torch.empty(0, dtype=torch.long, device=bar_embeddings.device))

        group_batch_indices = torch.tensor(group_batches, dtype=torch.long, device=bar_embeddings.device)
        return groups, group_batch_indices, group_token_indices

    def _scatter_prefix_outputs(
        self,
        *,
        all_prefixes: Tensor,
        group_batch_indices: Tensor,
        group_token_indices: list[Tensor],
        output_shape: torch.Size,
    ) -> Tensor:
        output = torch.zeros(output_shape, device=all_prefixes.device, dtype=all_prefixes.dtype)
        for group_index, token_indices in enumerate(group_token_indices):
            if token_indices.numel() == 0:
                continue
            batch_index = int(group_batch_indices[group_index].item())
            output[batch_index, token_indices] = all_prefixes[group_index, : token_indices.numel()]

        return output

    def _build_bar_memory_padding_mask(
        self,
        *,
        per_batch_vectors: list[Tensor],
        max_bars: int,
        device: torch.device,
    ) -> Tensor:
        bar_counts = torch.tensor([vectors.size(0) for vectors in per_batch_vectors], dtype=torch.long, device=device)
        bar_indices = torch.arange(max_bars, device=device).unsqueeze(0)
        return bar_indices >= bar_counts.unsqueeze(1)

    def _build_target_padding_mask(
        self,
        *,
        bar_positions: Tensor,
        token_padding_mask: Tensor | None,
    ) -> Tensor:
        inferred_padding_mask = bar_positions < 0
        if token_padding_mask is None:
            return inferred_padding_mask

        if token_padding_mask.shape != bar_positions.shape:
            token_padding_shape = tuple(token_padding_mask.shape)
            bar_positions_shape = tuple(bar_positions.shape)
            raise ValueError(
                f"token_padding_mask shape {token_padding_shape} does not match "
                f"bar_positions shape {bar_positions_shape}"
            )

        return inferred_padding_mask | token_padding_mask.to(device=bar_positions.device, dtype=torch.bool)

    def _build_memory_padding_mask(self, *, bar_memory_padding_mask: Tensor) -> Tensor:
        condition_padding_mask = torch.zeros(
            bar_memory_padding_mask.size(0),
            1,
            dtype=torch.bool,
            device=bar_memory_padding_mask.device,
        )
        return torch.cat([condition_padding_mask, bar_memory_padding_mask], dim=1)

    def _build_conditioning_memory_padding_mask(self, memory_context: Tensor) -> Tensor:
        return torch.zeros(
            memory_context.size(0),
            memory_context.size(1),
            dtype=torch.bool,
            device=memory_context.device,
        )

    def _build_conditioning_memory_attention_mask(self, *, bar_positions: Tensor) -> Tensor:
        batch_size, sequence_length = bar_positions.shape
        return torch.zeros(
            batch_size * self._config.transformer.num_heads,
            sequence_length,
            1,
            dtype=torch.bool,
            device=bar_positions.device,
        )

    def _build_memory_attention_mask(self, *, bar_positions: Tensor, num_bar_memory: int) -> Tensor:
        batch_size, sequence_length = bar_positions.shape
        condition_mask = torch.zeros(batch_size, sequence_length, 1, dtype=torch.bool, device=bar_positions.device)
        bar_indices = torch.arange(num_bar_memory, device=bar_positions.device).view(1, 1, num_bar_memory)
        visible_bars = (bar_indices < bar_positions.unsqueeze(-1)) & (bar_positions.unsqueeze(-1) >= 0)
        bar_mask = ~visible_bars
        memory_mask = torch.cat([condition_mask, bar_mask], dim=2)
        return memory_mask.repeat_interleave(self._config.transformer.num_heads, dim=0)

    def _encode_bars(self, *, bar_embeddings: Tensor, bar_positions: Tensor) -> Tensor:
        self._validate_bar_input_shapes(bar_embeddings=bar_embeddings, bar_positions=bar_positions)

        all_bar_groups, group_batch_indices = self._collect_all_bar_groups(
            bar_embeddings=bar_embeddings,
            bar_positions=bar_positions,
        )

        all_bar_vectors = self._run_gru_on_all_bars(all_bar_groups)  # [total_bars, H]

        per_batch_vectors = self._split_vectors_per_item(
            all_bar_vectors=all_bar_vectors,
            group_batch_indices=group_batch_indices,
            batch_size=bar_embeddings.size(0),
        )

        return self._pad_and_stack_to_batch(
            per_batch_vectors,
        )

    def _collect_all_bar_groups(
        self,
        *,
        bar_embeddings: Tensor,
        bar_positions: Tensor,
    ) -> tuple[list[Tensor], Tensor]:
        batch_size = bar_embeddings.size(0)
        max_bar_index = int(bar_positions.max().item())
        hidden_size = bar_embeddings.size(-1)
        fallback_result = self._build_batch_fallback_groups(
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=bar_embeddings.device,
            dtype=bar_embeddings.dtype,
        )

        if max_bar_index < 0:
            return fallback_result

        valid_tokens = self._extract_valid_bar_tokens(
            bar_embeddings=bar_embeddings,
            bar_positions=bar_positions,
            max_bar_index=max_bar_index,
        )
        if valid_tokens is None:
            return fallback_result

        filtered_embeddings, filtered_positions, filtered_batches = valid_tokens
        present_groups, present_batch_indices = self._group_tokens_by_batch_and_bar(
            filtered_embeddings=filtered_embeddings,
            filtered_positions=filtered_positions,
            filtered_batches=filtered_batches,
            max_bar_index=max_bar_index,
        )

        return self._append_missing_batch_fallbacks(
            present_groups=present_groups,
            present_batch_indices=present_batch_indices,
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=bar_embeddings.device,
            dtype=bar_embeddings.dtype,
        )

    def _build_batch_fallback_groups(
        self,
        *,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[list[Tensor], Tensor]:
        fallback_groups = [torch.zeros(1, hidden_size, device=device, dtype=dtype) for _ in range(batch_size)]
        fallback_batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
        return fallback_groups, fallback_batch_indices

    def _extract_valid_bar_tokens(
        self,
        *,
        bar_embeddings: Tensor,
        bar_positions: Tensor,
        max_bar_index: int,
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        valid_mask = (bar_positions >= 0) & (bar_positions <= max_bar_index)
        if not torch.any(valid_mask):
            return None

        batch_size = bar_positions.size(0)
        batch_indices = torch.arange(batch_size, device=bar_positions.device).unsqueeze(1).expand_as(bar_positions)
        filtered_embeddings = bar_embeddings[valid_mask]
        filtered_positions = bar_positions[valid_mask]
        filtered_batches = batch_indices[valid_mask]
        return filtered_embeddings, filtered_positions, filtered_batches

    def _group_tokens_by_batch_and_bar(
        self,
        *,
        filtered_embeddings: Tensor,
        filtered_positions: Tensor,
        filtered_batches: Tensor,
        max_bar_index: int,
    ) -> tuple[list[Tensor], Tensor]:
        stride = max_bar_index + 1
        global_bar_ids = filtered_batches * stride + filtered_positions
        order = torch.argsort(global_bar_ids, stable=True)

        sorted_global_bar_ids = global_bar_ids[order]
        sorted_embeddings = filtered_embeddings[order]
        unique_global_bar_ids, group_lengths = torch.unique_consecutive(sorted_global_bar_ids, return_counts=True)

        present_groups = list(torch.split(sorted_embeddings, group_lengths.tolist()))
        present_batch_indices = torch.div(unique_global_bar_ids, stride, rounding_mode="floor")
        return present_groups, present_batch_indices

    def _append_missing_batch_fallbacks(
        self,
        *,
        present_groups: list[Tensor],
        present_batch_indices: Tensor,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[list[Tensor], Tensor]:
        present_counts = torch.bincount(present_batch_indices, minlength=batch_size)
        missing_batch_mask = present_counts == 0

        if not torch.any(missing_batch_mask):
            return present_groups, present_batch_indices

        missing_batch_indices = torch.nonzero(missing_batch_mask, as_tuple=False).squeeze(1)
        fallback_groups = [
            torch.zeros(1, hidden_size, device=device, dtype=dtype) for _ in range(int(missing_batch_indices.numel()))
        ]
        all_groups = present_groups + fallback_groups
        all_batch_indices = torch.cat([present_batch_indices, missing_batch_indices], dim=0)
        return all_groups, all_batch_indices

    def _split_vectors_per_item(
        self,
        *,
        all_bar_vectors: Tensor,
        group_batch_indices: Tensor,
        batch_size: int,
    ) -> list[Tensor]:
        ordered_indices = torch.argsort(group_batch_indices, stable=True)
        ordered_vectors = all_bar_vectors[ordered_indices]
        ordered_batch_indices = group_batch_indices[ordered_indices]

        _, counts = torch.unique_consecutive(ordered_batch_indices, return_counts=True)
        split_vectors = list(torch.split(ordered_vectors, counts.tolist()))

        if len(split_vectors) != batch_size:
            raise ValueError(f"expected {batch_size} batch splits, got {len(split_vectors)}")

        return split_vectors

    def _validate_bar_input_shapes(self, *, bar_embeddings: Tensor, bar_positions: Tensor) -> None:
        if bar_positions.ndim != 2:
            raise ValueError(f"expected bar_positions with 2 dimensions, got {bar_positions.ndim}")

        if bar_positions.shape != bar_embeddings.shape[:2]:
            bar_positions_shape = tuple(bar_positions.shape)
            token_shape = tuple(bar_embeddings.shape[:2])
            raise ValueError(f"bar_positions shape {bar_positions_shape} does not match token shape {token_shape}")

    def _run_gru_on_all_bars(self, bar_groups: list[Tensor]) -> Tensor:
        # bar_groups: list of [bar_T_i, H] tensors with variable lengths
        # returns: [num_bars, H]
        lengths = torch.tensor([g.size(0) for g in bar_groups], dtype=torch.long)
        padded = pad_sequence(bar_groups, batch_first=True)  # [num_bars, max_bar_T, H]
        return cast(Tensor, self._bar_encoder(padded, lengths=lengths))  # [num_bars, H]

    def _pad_and_stack_to_batch(
        self,
        per_batch_vectors: list[Tensor],
    ) -> Tensor:
        # per_batch_vectors: list of [bars_i, H] tensors with variable bars_i
        # returns: [B, max_bars, H]
        return pad_sequence(per_batch_vectors, batch_first=True)

    def _build_conditioning_prefix(
        self,
        *,
        batch_size: int,
        device: torch.device,
        difficulty_ids: Tensor | None,
        scale_type_ids: Tensor | None,
        time_signature_ids: Tensor | None,
        structural_control_ids: Tensor | None,
    ) -> Tensor:
        conditioning = torch.zeros(
            batch_size,
            self._config.transformer.hidden_size,
            dtype=self._difficulty_embedding.weight.dtype,
            device=device,
        )

        difficulty_indices = self._to_optional_indices(
            input_ids=difficulty_ids,
            batch_size=batch_size,
            device=device,
            max_valid_id=self._config.conditioning.num_difficulty_levels,
            name="difficulty_ids",
        )
        scale_type_indices = self._to_optional_indices(
            input_ids=scale_type_ids,
            batch_size=batch_size,
            device=device,
            max_valid_id=self._config.conditioning.num_scale_types,
            name="scale_type_ids",
        )
        time_signature_indices = self._to_optional_indices(
            input_ids=time_signature_ids,
            batch_size=batch_size,
            device=device,
            max_valid_id=self._config.conditioning.num_time_signatures,
            name="time_signature_ids",
        )

        if difficulty_indices is not None:
            conditioning = conditioning + self._difficulty_embedding(difficulty_indices)

        if scale_type_indices is not None:
            conditioning = conditioning + self._scale_type_embedding(scale_type_indices)

        if time_signature_indices is not None:
            conditioning = conditioning + self._time_signature_embedding(time_signature_indices)

        if structural_control_ids is not None:
            conditioning = conditioning + self._structural_conditioning(
                structural_control_ids,
                batch_size=batch_size,
                device=device,
            )

        normalized_conditioning = cast(Tensor, self._conditioning_norm(conditioning))
        return normalized_conditioning.unsqueeze(1)

    def _structural_conditioning(
        self,
        structural_control_ids: Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if structural_control_ids.ndim != 2:
            raise ValueError(f"structural_control_ids must be 2D tensor, got {structural_control_ids.ndim}D")

        if structural_control_ids.size(0) != batch_size:
            raise ValueError(
                f"structural_control_ids batch size {structural_control_ids.size(0)} "
                f"does not match batch size {batch_size}"
            )

        if structural_control_ids.size(1) != len(self._structural_control_embeddings):
            raise ValueError(
                f"structural_control_ids width {structural_control_ids.size(1)} "
                f"does not match expected {len(self._structural_control_embeddings)}"
            )

        structural_ids = structural_control_ids.to(device=device, dtype=torch.long)
        conditioning = torch.zeros(
            batch_size,
            self._config.transformer.hidden_size,
            dtype=self._difficulty_embedding.weight.dtype,
            device=device,
        )
        for control_index, embedding in enumerate(self._structural_control_embeddings):
            structural_embedding = cast(nn.Embedding, embedding)
            control_ids = structural_ids[:, control_index]
            if torch.any(control_ids < 0):
                raise ValueError("structural_control_ids contains negative values")

            if torch.any(control_ids >= structural_embedding.num_embeddings):
                raise ValueError(
                    f"structural_control_ids column {control_index} contains values outside range "
                    f"[0, {structural_embedding.num_embeddings - 1}]"
                )

            conditioning = conditioning + structural_embedding(control_ids)

        return conditioning

    def _to_optional_indices(
        self,
        *,
        input_ids: Tensor | None,
        batch_size: int,
        device: torch.device,
        max_valid_id: int,
        name: str,
    ) -> Tensor | None:
        if input_ids is None:
            return None

        if input_ids.ndim != 1:
            raise ValueError(f"{name} must be 1D tensor, got {input_ids.ndim}D")

        if input_ids.size(0) != batch_size:
            raise ValueError(f"{name} length {input_ids.size(0)} does not match batch size {batch_size}")

        if torch.any(input_ids < 0):
            raise ValueError(f"{name} contains negative values")

        if torch.any(input_ids >= max_valid_id):
            raise ValueError(f"{name} contains values outside range [0, {max_valid_id - 1}]")

        return input_ids.to(device=device, dtype=torch.long)
