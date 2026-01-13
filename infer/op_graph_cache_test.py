from __future__ import annotations

import torch
from torch import nn

from caramba.config.layer import LayerType, OpGraphCacheFieldConfig, OpGraphLayerConfig
from caramba.config.kvcache import KVCacheKind
from caramba.infer.context import InferContext
from caramba.infer.generate import create_caches


def test_create_caches_includes_op_graph_layer_and_ctx_consumes_it() -> None:
    cfg = OpGraphLayerConfig(
        type=LayerType.OP_GRAPH,
        d_in=4,
        d_out=4,
        input_key="x",
        output_key="y",
        cache_fields=[
            OpGraphCacheFieldConfig(name="k", dim=4),
            OpGraphCacheFieldConfig(name="v", dim=4),
        ],
        graph={
            "nodes": [
                {"id": "next", "op": "InferCtxNextCacheOperation", "in": "infer_ctx", "out": "cache", "config": {}},
                # Append k/v (use x for both) then read the full cache back.
                {"id": "write", "op": "KVCacheWriteOperation", "in": ["cache", "x", "x"], "out": ["cache2", "old"], "config": {}},
                {"id": "read", "op": "KVCacheReadOperation", "in": "cache2", "out": ["a", "b"], "config": {}},
                {"id": "add", "op": "AddOperation", "in": ["a", "b"], "out": "y", "config": {}},
            ],
            "inputs": ["x"],
        },
    )
    op_layer = cfg.build()

    class M(nn.Module):
        def __init__(self, layer: nn.Module) -> None:
            super().__init__()
            self.layer = layer

        def forward(self, x: torch.Tensor, *, ctx: object | None = None) -> torch.Tensor:
            return self.layer(x, ctx=ctx)  # type: ignore[call-arg]

    model = M(op_layer)

    caches = create_caches(
        model,
        batch_size=1,
        max_seq_len=8,
        device=torch.device("cpu"),
        cache_kind=KVCacheKind.FP16,
        cache_qblock=32,
        cache_residual_len=0,
    )
    assert len(caches) == 1

    ctx = InferContext(caches=caches)
    ctx.begin(pos_offset=0)
    x = torch.randn(1, 3, 4)
    y = model(x, ctx=ctx)
    ctx.ensure_consumed()

    assert y.shape == x.shape
    assert torch.allclose(y, (x + x).to(dtype=y.dtype))
