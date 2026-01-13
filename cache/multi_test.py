from __future__ import annotations

import pytest
import torch

from caramba.cache.multi import CacheFieldSpec, MultiKVCache
from caramba.config.kvcache import KVCacheKind, KVCacheTensorConfig


def test_multi_kvcache_append_get_and_truncate() -> None:
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = MultiKVCache(
        batch_size=2,
        max_seq_len=8,
        fields=[
            CacheFieldSpec(name="k", dim=6, cfg=cfg),
            CacheFieldSpec(name="v", dim=6, cfg=cfg),
        ],
        device=torch.device("cpu"),
    )

    k0 = torch.randn(2, 3, 6, dtype=torch.float32)
    v0 = torch.randn(2, 3, 6, dtype=torch.float32)
    old = cache.append_many({"k": k0, "v": v0})
    assert old == 0
    assert cache.pos == 3

    got = cache.get_many(dtype=torch.float32)
    assert set(got.keys()) == {"k", "v"}
    assert got["k"].shape == (2, 3, 6)
    assert got["v"].shape == (2, 3, 6)

    cache.truncate(1)
    assert cache.pos == 1
    got2 = cache.get_many(dtype=torch.float32)
    assert got2["k"].shape == (2, 1, 6)


def test_multi_kvcache_validates_field_names_and_append_keys() -> None:
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)

    with pytest.raises(ValueError, match="at least one field"):
        _ = MultiKVCache(batch_size=1, max_seq_len=4, fields=[], device=torch.device("cpu"))

    with pytest.raises(ValueError, match="unique"):
        _ = MultiKVCache(
            batch_size=1,
            max_seq_len=4,
            fields=[
                CacheFieldSpec(name="k", dim=4, cfg=cfg),
                CacheFieldSpec(name="k", dim=4, cfg=cfg),
            ],
            device=torch.device("cpu"),
        )

    cache = MultiKVCache(
        batch_size=1,
        max_seq_len=4,
        fields=[CacheFieldSpec(name="k", dim=4, cfg=cfg), CacheFieldSpec(name="v", dim=4, cfg=cfg)],
        device=torch.device("cpu"),
    )
    k0 = torch.randn(1, 1, 4)
    v0 = torch.randn(1, 1, 4)

    with pytest.raises(KeyError, match="append_many mismatch"):
        cache.append_many({"k": k0})

    with pytest.raises(KeyError, match="append_many mismatch"):
        cache.append_many({"k": k0, "v": v0, "extra": k0})

