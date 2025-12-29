"""High-performance Recurrent Neural Network layer.

Wraps Torch's cuDNN-optimized RNN implementations (LSTM/GRU) with
Caramba's configuration interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

import torch
from torch import Tensor, nn
from typing_extensions import override

from config.layer import RNNLayerConfig

if TYPE_CHECKING:
    from tensordict import TensorDictBase


class RNNLayer(nn.Module):
    """Recurrent Neural Network layer (LSTM/GRU).
    
    Uses PyTorch's native implementations which dispatch to fused cuDNN kernels
    on GPU for maximum performance.
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
        """Apply RNN.
        
        Args:
            x: Input tensor (B, T, input_size) if batch_first=True
            ctx: Context (can contain initial hidden state 'h0' or 'c0')
            
        Returns:
            output: (B, T, D * hidden_size)
            hidden: final hidden state
        """
        # TODO: Extract h0/c0 from ctx if present
        return self.rnn(x)
