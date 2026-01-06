"""Recurrent neural network layer

RNNs are the original sequence models: they update a hidden state over time,
which can be a useful inductive bias when you want explicit recurrence instead
of attention.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import torch
from torch import Tensor, nn
from typing_extensions import override

from caramba.config.layer import RNNLayerConfig

if TYPE_CHECKING:
    from tensordict import TensorDictBase


class RNNLayer(nn.Module):
    """Recurrent neural network layer

    This is a thin wrapper around PyTorch's optimized RNN kernels, so you can
    include LSTM/GRU-style recurrence in a manifest-driven topology.
    """

    def __init__(self, config: RNNLayerConfig) -> None:
        super().__init__()
        self.config = config
        
        if config.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                bias=config.bias,
                batch_first=config.batch_first,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                bidirectional=config.bidirectional,
                proj_size=config.proj_size,
            )
        elif config.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                bias=config.bias,
                batch_first=config.batch_first,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                bidirectional=config.bidirectional,
            )
        elif config.cell_type == "rnn_tanh":
            self.rnn = nn.RNN(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                nonlinearity="tanh",
                bias=config.bias,
                batch_first=config.batch_first,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                bidirectional=config.bidirectional,
            )
        elif config.cell_type == "rnn_relu":
            self.rnn = nn.RNN(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                nonlinearity="relu",
                bias=config.bias,
                batch_first=config.batch_first,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                bidirectional=config.bidirectional,
            )
        else:
            raise ValueError(f"Unknown cell_type: {config.cell_type}")

    @override
    def forward(
        self,
        x: Tensor,
        *,
        ctx: object | None = None,
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        """Apply recurrence

        RNNs compress history into a fixed-size state; that can be a feature when
        you want bounded memory rather than full-context attention.
        """
        # TODO: Extract h0/c0 from ctx if present
        return self.rnn(x)
