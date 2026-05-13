import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from musak_model.model.config import CNNConfig


class ResidualConvBlock(nn.Module):
    def __init__(self, *, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self._conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self._conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self._norm1 = nn.LayerNorm(channels)
        self._norm2 = nn.LayerNorm(channels)
        self._dropout = nn.Dropout(dropout)

    def forward(self, embeddings: Tensor) -> Tensor:
        residual = embeddings

        # (batch, channels, seq) — transpose in/out
        out = self._norm1(embeddings.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(self._conv1(out))  # pylint: disable=not-callable
        out = self._dropout(out)
        out = self._norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self._conv2(out)
        out = self._dropout(out)
        return F.gelu(out + residual)  # pylint: disable=not-callable


class MultiKernelConvLayer(nn.Module):
    def __init__(self, *, channels: int, kernel_sizes: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        self._branches = nn.ModuleList(
            [
                ResidualConvBlock(channels=channels, kernel_size=kernel_size, dropout=dropout)
                for kernel_size in kernel_sizes
            ]
        )
        self._mix = nn.Linear(channels * len(kernel_sizes), channels)

    def forward(self, embeddings: Tensor) -> Tensor:
        branch_outputs = [branch(embeddings) for branch in self._branches]
        # branch_outputs: list of (batch, channels, seq) — concat along channels
        concatenated = torch.cat(branch_outputs, dim=1)
        mixed = self._mix(concatenated.transpose(1, 2)).transpose(1, 2)
        return F.gelu(mixed)  # pylint: disable=not-callable


class LocalConvEncoder(nn.Module):
    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self._layers = nn.ModuleList(
            [
                MultiKernelConvLayer(
                    channels=config.out_channels,
                    kernel_sizes=config.kernel_sizes,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        # embeddings: (batch, seq, hidden)
        # conv layers expect (batch, channels, seq)
        out = embeddings.transpose(1, 2)
        for layer in self._layers:
            out = layer(out)

        return out.transpose(1, 2)
