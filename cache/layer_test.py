from __future__ import annotations

import pytest
import torch

from caramba.cache.layer import LayerKVCache
from caramba.config.kvcache import KVCacheKind, KVCacheTensorConfig


def test_layer_kv_cache_init() -> None:
    """Test LayerKVCache initialization with proper tensor allocation."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = LayerKVCache(
        batch_size=2,
        max_seq_len=8,
        k_dim=4,
        v_dim=6,
        k_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    assert cache.pos == 0
    assert cache.keys == ("k", "v")


def test_layer_kv_cache_append() -> None:
    """Test appending new K and V tensors."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = LayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_dim=4,
        v_dim=6,
        k_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k = torch.randn(1, 3, 4, dtype=torch.float32)
    v = torch.randn(1, 3, 6, dtype=torch.float32)

    old_pos = cache.append(k, v)
    assert old_pos == 0
    assert cache.pos == 3


def test_layer_kv_cache_append_position_mismatch() -> None:
    """Test that append fails when K and V caches have position mismatches."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = LayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_dim=4,
        v_dim=6,
        k_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    # First append some data to establish positions
    k1 = torch.randn(1, 2, 4, dtype=torch.float32)
    v1 = torch.randn(1, 2, 6, dtype=torch.float32)
    cache.append(k1, v1)
    assert cache.pos == 2

    # Manually mess with one cache's position to simulate internal inconsistency
    cache.k.pos = 3  # type: ignore[attr-defined]

    k2 = torch.randn(1, 1, 4, dtype=torch.float32)
    v2 = torch.randn(1, 1, 6, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="K/V append position mismatch"):
        cache.append(k2, v2)


def test_layer_kv_cache_get() -> None:
    """Test retrieving all cached K and V tensors."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = LayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_dim=4,
        v_dim=6,
        k_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k = torch.randn(1, 3, 4, dtype=torch.float32)
    v = torch.randn(1, 3, 6, dtype=torch.float32)

    cache.append(k, v)
    k_got, v_got = cache.get(dtype=torch.float32)

    assert k_got.shape == (1, 3, 4)
    assert v_got.shape == (1, 3, 6)

    # Check that data is stored and retrieved correctly (accounting for fp16 precision)
    assert torch.allclose(k_got, k.to(torch.float16).to(torch.float32), atol=1e-4)
    assert torch.allclose(v_got, v.to(torch.float16).to(torch.float32), atol=1e-4)


def test_layer_kv_cache_get_slice() -> None:
    """Test retrieving a slice of cached tokens."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = LayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_dim=4,
        v_dim=6,
        k_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k = torch.randn(1, 5, 4, dtype=torch.float32)
    v = torch.randn(1, 5, 6, dtype=torch.float32)

    cache.append(k, v)
    k_slice, v_slice = cache.get_slice(1, 4, dtype=torch.float32)

    assert k_slice.shape == (1, 3, 4)
    assert v_slice.shape == (1, 3, 6)

    # Check slice content
    assert torch.allclose(k_slice, k[:, 1:4].to(torch.float16).to(torch.float32), atol=1e-4)
    assert torch.allclose(v_slice, v[:, 1:4].to(torch.float16).to(torch.float32), atol=1e-4)


def test_layer_kv_cache_get_slice_bounds() -> None:
    """Test slice bounds validation."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = LayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_dim=4,
        v_dim=6,
        k_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k = torch.randn(1, 4, 4, dtype=torch.float32)
    v = torch.randn(1, 4, 6, dtype=torch.float32)
    cache.append(k, v)

    # Valid slice
    cache.get_slice(1, 3)

    # Invalid slices
    with pytest.raises(ValueError, match="Invalid slice"):
        cache.get_slice(3, 1)  # start > end

    with pytest.raises(ValueError, match="Requested end"):
        cache.get_slice(0, 6)  # end > pos


def test_layer_kv_cache_truncate() -> None:
    """Test truncating cache to a previous position."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = LayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_dim=4,
        v_dim=6,
        k_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k = torch.randn(1, 5, 4, dtype=torch.float32)
    v = torch.randn(1, 5, 6, dtype=torch.float32)

    cache.append(k, v)
    assert cache.pos == 5

    cache.truncate(2)
    assert cache.pos == 2

    # Check that get returns truncated data
    k_got, v_got = cache.get(dtype=torch.float32)
    assert k_got.shape == (1, 2, 4)
    assert v_got.shape == (1, 2, 6)


def test_layer_kv_cache_truncate_to_valid_position() -> None:
    """Test truncating cache to a valid previous position."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = LayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_dim=4,
        v_dim=6,
        k_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    # First append some data
    k = torch.randn(1, 3, 4, dtype=torch.float32)
    v = torch.randn(1, 3, 6, dtype=torch.float32)
    cache.append(k, v)
    assert cache.pos == 3

    # Truncate to a valid position
    cache.truncate(1)
    assert cache.pos == 1

    # Check that get returns truncated data
    k_got, v_got = cache.get(dtype=torch.float32)
    assert k_got.shape == (1, 1, 4)
    assert v_got.shape == (1, 1, 6)


def test_layer_kv_cache_append_many() -> None:
    """Test the generic append_many API."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = LayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_dim=4,
        v_dim=6,
        k_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k = torch.randn(1, 2, 4, dtype=torch.float32)
    v = torch.randn(1, 2, 6, dtype=torch.float32)

    old_pos = cache.append_many({"k": k, "v": v})
    assert old_pos == 0
    assert cache.pos == 2


def test_layer_kv_cache_append_many_wrong_keys() -> None:
    """Test append_many validation with wrong keys."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = LayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_dim=4,
        v_dim=6,
        k_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    # Missing key
    with pytest.raises(KeyError, match="requires keys"):
        cache.append_many({"k": torch.randn(1, 1, 4)})

    # Extra key
    with pytest.raises(KeyError, match="mismatch.*extra.*extra"):
        cache.append_many({
            "k": torch.randn(1, 1, 4),
            "v": torch.randn(1, 1, 6),
            "extra": torch.randn(1, 1, 2),
        })

    # Wrong key name
    with pytest.raises(KeyError, match="requires keys"):
        cache.append_many({
            "k": torch.randn(1, 1, 4),
            "wrong": torch.randn(1, 1, 6),
        })


def test_layer_kv_cache_get_many() -> None:
    """Test the generic get_many API."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = LayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_dim=4,
        v_dim=6,
        k_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k = torch.randn(1, 3, 4, dtype=torch.float32)
    v = torch.randn(1, 3, 6, dtype=torch.float32)
    cache.append_many({"k": k, "v": v})

    got = cache.get_many(dtype=torch.float32)
    assert set(got.keys()) == {"k", "v"}
    assert got["k"].shape == (1, 3, 4)
    assert got["v"].shape == (1, 3, 6)


def test_layer_kv_cache_get_slice_many() -> None:
    """Test the generic get_slice_many API."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = LayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_dim=4,
        v_dim=6,
        k_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k = torch.randn(1, 5, 4, dtype=torch.float32)
    v = torch.randn(1, 5, 6, dtype=torch.float32)
    cache.append_many({"k": k, "v": v})

    got = cache.get_slice_many(1, 4, dtype=torch.float32)
    assert set(got.keys()) == {"k", "v"}
    assert got["k"].shape == (1, 3, 4)
    assert got["v"].shape == (1, 3, 6)


def test_layer_kv_cache_different_configurations() -> None:
    """Test LayerKVCache with different batch sizes, sequence lengths, and dimensions."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)

    # Test different batch sizes
    for batch_size in [1, 2, 4]:
        cache = LayerKVCache(
            batch_size=batch_size,
            max_seq_len=8,
            k_dim=4,
            v_dim=6,
            k_cfg=cfg,
            v_cfg=cfg,
            device=torch.device("cpu"),
        )

        k = torch.randn(batch_size, 2, 4, dtype=torch.float32)
        v = torch.randn(batch_size, 2, 6, dtype=torch.float32)
        cache.append(k, v)
        assert cache.pos == 2

    # Test different dimensions
    for k_dim, v_dim in [(8, 8), (16, 32), (32, 64)]:
        cache = LayerKVCache(
            batch_size=1,
            max_seq_len=8,
            k_dim=k_dim,
            v_dim=v_dim,
            k_cfg=cfg,
            v_cfg=cfg,
            device=torch.device("cpu"),
        )

        k = torch.randn(1, 1, k_dim, dtype=torch.float32)
        v = torch.randn(1, 1, v_dim, dtype=torch.float32)
        cache.append(k, v)

        k_got, v_got = cache.get(dtype=torch.float32)
        assert k_got.shape == (1, 1, k_dim)
        assert v_got.shape == (1, 1, v_dim)