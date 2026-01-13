"""Tests for KVCacheReadOperation (cache object semantics)."""
from __future__ import annotations

import unittest

import torch

from caramba.cache.layer import LayerKVCache
from caramba.cache.multi import CacheFieldSpec, MultiKVCache
from caramba.config.kvcache import KVCacheKind, KVCacheTensorConfig
from caramba.operation.cache.kv_cache_read import KVCacheReadOperation
from caramba.operation.cache.kv_cache_write import KVCacheWriteOperation


class TestKVCacheReadOperation(unittest.TestCase):
    def test_reads_slice_from_standard_layer_cache(self) -> None:
        cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
        cache = LayerKVCache(
            batch_size=1,
            max_seq_len=8,
            k_dim=4,
            v_dim=4,
            k_cfg=cfg,
            v_cfg=cfg,
            device=torch.device("cpu"),
        )
        write = KVCacheWriteOperation()
        read = KVCacheReadOperation()

        k = torch.arange(1 * 5 * 4, dtype=torch.float16).view(1, 5, 4)
        v = torch.arange(1 * 5 * 4, 2 * 1 * 5 * 4, dtype=torch.float16).view(1, 5, 4)
        _ = write(cache, k, v)

        k2, v2 = read(cache, start_pos=2, seq_len=2, dtype=torch.float16)  # type: ignore[misc]
        self.assertTrue(torch.allclose(k2, k[:, 2:4]))
        self.assertTrue(torch.allclose(v2, v[:, 2:4]))

    def test_reads_slice_from_multi_kv_cache(self) -> None:
        cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
        cache = MultiKVCache(
            batch_size=1,
            max_seq_len=8,
            fields=[
                CacheFieldSpec(name="a", dim=2, cfg=cfg),
                CacheFieldSpec(name="b", dim=3, cfg=cfg),
            ],
            device=torch.device("cpu"),
        )
        write = KVCacheWriteOperation()
        read = KVCacheReadOperation(keys=["b"])

        a = torch.randn(1, 4, 2, dtype=torch.float16)
        b = torch.randn(1, 4, 3, dtype=torch.float16)
        _ = write(cache, a, b)

        b_slice = read(cache, start_pos=1, seq_len=2, dtype=torch.float16)
        self.assertIsInstance(b_slice, torch.Tensor)
        self.assertTrue(torch.allclose(b_slice, b[:, 1:3]))

