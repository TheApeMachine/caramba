from __future__ import annotations

from pathlib import Path

import torch
from pydantic import TypeAdapter

from caramba.cache.multi import CacheFieldSpec, MultiKVCache
from caramba.config.kvcache import KVCacheTensorConfig
from caramba.config.layer import LayerConfig
from caramba.config.resolve import Resolver
from caramba.config.yaml_include import load_yaml
from caramba.infer.context import InferContext


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_layer_from_block(path: Path, *, vars: dict[str, object]):
    payload = load_yaml(path)
    resolved = Resolver(vars).resolve(payload)
    cfg = TypeAdapter(LayerConfig).validate_python(resolved)
    return cfg.build()


def test_standard_opgraph_block_builds_and_runs() -> None:
    layer = _build_layer_from_block(
        _repo_root() / "config/blocks/attention/standard_opgraph.yml",
        vars={
            "d_model": 8,
            "n_heads": 2,
            "n_kv_heads": 1,
            "group_size": 2,
            "q_dim": 8,
            "kv_dim": 4,
            "qkv_dim": 16,
            "rope_base": 10000.0,
            "dropout_p": 0.0,
            "is_causal": True,
        },
    )

    x = torch.randn(2, 3, 8)
    y = layer(x)
    assert y.shape == x.shape


def test_dba_opgraph_null_block_builds_and_runs() -> None:
    layer = _build_layer_from_block(
        _repo_root() / "config/blocks/attention/dba_opgraph_null.yml",
        vars={
            "d_model": 8,
            "n_heads": 2,
            "sem_dim": 4,
            "geo_dim": 4,
            "attn_dim": 8,
            "rope_base": 10000.0,
            "dropout_p": 0.0,
        },
    )

    x = torch.randn(2, 3, 8)
    y = layer(x)
    assert y.shape == x.shape


def test_standard_opgraph_cache_block_runs_with_infer_context() -> None:
    layer = _build_layer_from_block(
        _repo_root() / "config/blocks/attention/standard_opgraph_cache.yml",
        vars={
            "d_model": 8,
            "n_heads": 2,
            "n_kv_heads": 1,
            "group_size": 2,
            "q_dim": 8,
            "kv_dim": 4,
            "qkv_dim": 16,
            "rope_base": 10000.0,
            "dropout_p": 0.0,
        },
    ).to(dtype=torch.float16)

    B = 2
    cache = MultiKVCache(
        batch_size=B,
        max_seq_len=8,
        fields=[
            CacheFieldSpec(name="k", dim=4, cfg=KVCacheTensorConfig()),
            CacheFieldSpec(name="v", dim=4, cfg=KVCacheTensorConfig()),
        ],
        device=torch.device("cpu"),
    )
    ctx = InferContext(caches=[cache])

    # Decode step 0
    x0 = torch.randn(B, 1, 8, dtype=torch.float16)
    ctx.begin(pos_offset=0)
    y0 = layer(x0, ctx=ctx)
    ctx.ensure_consumed()
    assert y0.shape == x0.shape
    assert cache.pos == 1

    # Decode step 1 (pos_offset advances)
    x1 = torch.randn(B, 1, 8, dtype=torch.float16)
    ctx.begin(pos_offset=1)
    y1 = layer(x1, ctx=ctx)
    ctx.ensure_consumed()
    assert y1.shape == x1.shape
    assert cache.pos == 2

