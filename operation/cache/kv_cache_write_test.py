"""Tests for KVCacheWriteOperation (append semantics)."""
from __future__ import annotations

import unittest

import torch

from caramba.cache.decoupled import DecoupledLayerKVCache
from caramba.cache.layer import LayerKVCache
from caramba.cache.multi import CacheFieldSpec, MultiKVCache
from caramba.config.kvcache import KVCacheKind, KVCacheTensorConfig
from caramba.operation.cache.kv_cache_write import KVCacheWriteOperation


class TestKVCacheWriteOperation(unittest.TestCase):
    def test_appends_to_standard_layer_cache(self) -> None:
        cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
        cache = LayerKVCache(
            batch_size=2,
            max_seq_len=8,
            k_dim=4,
            v_dim=4,
            k_cfg=cfg,
            v_cfg=cfg,
            device=torch.device("cpu"),
        )
        op = KVCacheWriteOperation()

        k0 = torch.randn(2, 3, 4, dtype=torch.float16)
        v0 = torch.randn(2, 3, 4, dtype=torch.float16)
        cache0, old0 = op(cache, k0, v0)
        self.assertIs(cache0, cache)
        self.assertEqual(old0, 0)
        self.assertEqual(cache.pos, 3)

        k1 = torch.randn(2, 2, 4, dtype=torch.float16)
        v1 = torch.randn(2, 2, 4, dtype=torch.float16)
        cache1, old1 = op(cache, k1, v1)
        self.assertIs(cache1, cache)
        self.assertEqual(old1, 3)
        self.assertEqual(cache.pos, 5)

    def test_appends_to_decoupled_layer_cache(self) -> None:
        cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
        cache = DecoupledLayerKVCache(
            batch_size=1,
            max_seq_len=8,
            k_sem_dim=6,
            k_geo_dim=8,
            v_dim=10,
            k_sem_cfg=cfg,
            k_geo_cfg=cfg,
            v_cfg=cfg,
            device=torch.device("cpu"),
        )
        op = KVCacheWriteOperation()

        k_sem = torch.randn(1, 4, 6, dtype=torch.float16)
        k_geo = torch.randn(1, 4, 8, dtype=torch.float16)
        v = torch.randn(1, 4, 10, dtype=torch.float16)
        cache0, old = op(cache, k_sem, k_geo, v)
        self.assertIs(cache0, cache)
        self.assertEqual(old, 0)
        self.assertEqual(cache.pos, 4)

    def test_appends_to_multi_kv_cache(self) -> None:
        cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
        cache = MultiKVCache(
            batch_size=1,
            max_seq_len=8,
            fields=[
                CacheFieldSpec(name="a", dim=3, cfg=cfg),
                CacheFieldSpec(name="b", dim=5, cfg=cfg),
            ],
            device=torch.device("cpu"),
        )
        op = KVCacheWriteOperation()

        a = torch.randn(1, 2, 3, dtype=torch.float16)
        b = torch.randn(1, 2, 5, dtype=torch.float16)
        cache0, old = op(cache, a, b)
        self.assertIs(cache0, cache)
        self.assertEqual(old, 0)
        self.assertEqual(cache.pos, 2)

    def test_validates_tensor_count(self) -> None:
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
        op = KVCacheWriteOperation()
        k = torch.randn(1, 1, 4, dtype=torch.float16)
        with self.assertRaises(ValueError):
            _ = op(cache, k)
