from __future__ import annotations

import pytest
import torch

from caramba.cache.decoupled import DecoupledLayerKVCache
from caramba.config.kvcache import KVCacheKind, KVCacheTensorConfig


def test_decoupled_layer_kv_cache_init() -> None:
    """Test DecoupledLayerKVCache initialization with proper tensor allocation."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = DecoupledLayerKVCache(
        batch_size=2,
        max_seq_len=8,
        k_sem_dim=6,
        k_geo_dim=8,
        v_dim=10,
        k_sem_cfg=cfg,
        k_geo_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    assert cache.pos == 0
    assert cache.keys == ("k_sem", "k_geo", "v")


def test_decoupled_layer_kv_cache_append() -> None:
    """Test appending new tokens to decoupled cache."""
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

    k_sem = torch.randn(1, 3, 6, dtype=torch.float32)
    k_geo = torch.randn(1, 3, 8, dtype=torch.float32)
    v = torch.randn(1, 3, 10, dtype=torch.float32)

    old_pos = cache.append(k_sem, k_geo, v)
    assert old_pos == 0
    assert cache.pos == 3


def test_decoupled_layer_kv_cache_append_position_mismatch() -> None:
    """Test that append fails when internal caches have position mismatches."""
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

    # First append some data to establish positions
    k_sem1 = torch.randn(1, 2, 6, dtype=torch.float32)
    k_geo1 = torch.randn(1, 2, 8, dtype=torch.float32)
    v1 = torch.randn(1, 2, 10, dtype=torch.float32)
    cache.append(k_sem1, k_geo1, v1)
    assert cache.pos == 2

    # Manually mess with one cache's position to simulate internal inconsistency
    cache.k_sem.pos = 3  # type: ignore[attr-defined]

    k_sem2 = torch.randn(1, 1, 6, dtype=torch.float32)
    k_geo2 = torch.randn(1, 1, 8, dtype=torch.float32)
    v2 = torch.randn(1, 1, 10, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="Decoupled K/V append position mismatch"):
        cache.append(k_sem2, k_geo2, v2)


def test_decoupled_layer_kv_cache_get() -> None:
    """Test retrieving all cached tokens."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = DecoupledLayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_sem_dim=4,
        k_geo_dim=6,
        v_dim=8,
        k_sem_cfg=cfg,
        k_geo_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k_sem = torch.randn(1, 3, 4, dtype=torch.float32)
    k_geo = torch.randn(1, 3, 6, dtype=torch.float32)
    v = torch.randn(1, 3, 8, dtype=torch.float32)

    cache.append(k_sem, k_geo, v)
    k_sem_got, k_geo_got, v_got = cache.get(dtype=torch.float32)

    assert k_sem_got.shape == (1, 3, 4)
    assert k_geo_got.shape == (1, 3, 6)
    assert v_got.shape == (1, 3, 8)

    # Check that data is stored and retrieved correctly (accounting for fp16 precision)
    assert torch.allclose(k_sem_got, k_sem.to(torch.float16).to(torch.float32), atol=1e-4)
    assert torch.allclose(k_geo_got, k_geo.to(torch.float16).to(torch.float32), atol=1e-4)
    assert torch.allclose(v_got, v.to(torch.float16).to(torch.float32), atol=1e-4)


def test_decoupled_layer_kv_cache_get_slice() -> None:
    """Test retrieving a slice of cached tokens."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = DecoupledLayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_sem_dim=4,
        k_geo_dim=6,
        v_dim=8,
        k_sem_cfg=cfg,
        k_geo_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k_sem = torch.randn(1, 5, 4, dtype=torch.float32)
    k_geo = torch.randn(1, 5, 6, dtype=torch.float32)
    v = torch.randn(1, 5, 8, dtype=torch.float32)

    cache.append(k_sem, k_geo, v)
    k_sem_slice, k_geo_slice, v_slice = cache.get_slice(1, 4, dtype=torch.float32)

    assert k_sem_slice.shape == (1, 3, 4)
    assert k_geo_slice.shape == (1, 3, 6)
    assert v_slice.shape == (1, 3, 8)

    # Check slice content
    assert torch.allclose(k_sem_slice, k_sem[:, 1:4].to(torch.float16).to(torch.float32), atol=1e-4)
    assert torch.allclose(k_geo_slice, k_geo[:, 1:4].to(torch.float16).to(torch.float32), atol=1e-4)
    assert torch.allclose(v_slice, v[:, 1:4].to(torch.float16).to(torch.float32), atol=1e-4)


def test_decoupled_layer_kv_cache_get_slice_bounds() -> None:
    """Test slice bounds validation."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = DecoupledLayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_sem_dim=4,
        k_geo_dim=6,
        v_dim=8,
        k_sem_cfg=cfg,
        k_geo_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k_sem = torch.randn(1, 4, 4, dtype=torch.float32)
    k_geo = torch.randn(1, 4, 6, dtype=torch.float32)
    v = torch.randn(1, 4, 8, dtype=torch.float32)
    cache.append(k_sem, k_geo, v)

    # Valid slice
    cache.get_slice(1, 3)

    # Invalid slices
    with pytest.raises(ValueError, match="Invalid slice"):
        cache.get_slice(3, 1)  # start > end

    with pytest.raises(ValueError, match="Requested end"):
        cache.get_slice(0, 6)  # end > pos


def test_decoupled_layer_kv_cache_truncate() -> None:
    """Test truncating cache to a previous position."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = DecoupledLayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_sem_dim=4,
        k_geo_dim=6,
        v_dim=8,
        k_sem_cfg=cfg,
        k_geo_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k_sem = torch.randn(1, 5, 4, dtype=torch.float32)
    k_geo = torch.randn(1, 5, 6, dtype=torch.float32)
    v = torch.randn(1, 5, 8, dtype=torch.float32)

    cache.append(k_sem, k_geo, v)
    assert cache.pos == 5

    cache.truncate(2)
    assert cache.pos == 2

    # Check that get returns truncated data
    k_sem_got, k_geo_got, v_got = cache.get(dtype=torch.float32)
    assert k_sem_got.shape == (1, 2, 4)
    assert k_geo_got.shape == (1, 2, 6)
    assert v_got.shape == (1, 2, 8)


def test_decoupled_layer_kv_cache_truncate_to_valid_position() -> None:
    """Test truncating cache to a valid previous position."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = DecoupledLayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_sem_dim=4,
        k_geo_dim=6,
        v_dim=8,
        k_sem_cfg=cfg,
        k_geo_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    # First append some data
    k_sem = torch.randn(1, 3, 4, dtype=torch.float32)
    k_geo = torch.randn(1, 3, 6, dtype=torch.float32)
    v = torch.randn(1, 3, 8, dtype=torch.float32)
    cache.append(k_sem, k_geo, v)
    assert cache.pos == 3

    # Truncate to a valid position
    cache.truncate(1)
    assert cache.pos == 1

    # Check that get returns truncated data
    k_sem_got, k_geo_got, v_got = cache.get(dtype=torch.float32)
    assert k_sem_got.shape == (1, 1, 4)
    assert k_geo_got.shape == (1, 1, 6)
    assert v_got.shape == (1, 1, 8)


def test_decoupled_layer_kv_cache_append_many() -> None:
    """Test the generic append_many API."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = DecoupledLayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_sem_dim=4,
        k_geo_dim=6,
        v_dim=8,
        k_sem_cfg=cfg,
        k_geo_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k_sem = torch.randn(1, 2, 4, dtype=torch.float32)
    k_geo = torch.randn(1, 2, 6, dtype=torch.float32)
    v = torch.randn(1, 2, 8, dtype=torch.float32)

    old_pos = cache.append_many({"k_sem": k_sem, "k_geo": k_geo, "v": v})
    assert old_pos == 0
    assert cache.pos == 2


def test_decoupled_layer_kv_cache_append_many_wrong_keys() -> None:
    """Test append_many validation with wrong keys."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = DecoupledLayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_sem_dim=4,
        k_geo_dim=6,
        v_dim=8,
        k_sem_cfg=cfg,
        k_geo_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    # Missing key
    with pytest.raises(KeyError, match="missing.*k_sem"):
        cache.append_many({"k_geo": torch.randn(1, 1, 6), "v": torch.randn(1, 1, 8)})

    # Extra key
    with pytest.raises(KeyError, match="extra.*extra"):
        cache.append_many({
            "k_sem": torch.randn(1, 1, 4),
            "k_geo": torch.randn(1, 1, 6),
            "v": torch.randn(1, 1, 8),
            "extra": torch.randn(1, 1, 2),
        })

    # Wrong key name
    with pytest.raises(KeyError, match="mismatch.*missing.*v.*extra.*wrong"):
        cache.append_many({
            "k_sem": torch.randn(1, 1, 4),
            "k_geo": torch.randn(1, 1, 6),
            "wrong": torch.randn(1, 1, 8),
        })


def test_decoupled_layer_kv_cache_get_many() -> None:
    """Test the generic get_many API."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = DecoupledLayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_sem_dim=4,
        k_geo_dim=6,
        v_dim=8,
        k_sem_cfg=cfg,
        k_geo_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k_sem = torch.randn(1, 3, 4, dtype=torch.float32)
    k_geo = torch.randn(1, 3, 6, dtype=torch.float32)
    v = torch.randn(1, 3, 8, dtype=torch.float32)
    cache.append_many({"k_sem": k_sem, "k_geo": k_geo, "v": v})

    got = cache.get_many(dtype=torch.float32)
    assert set(got.keys()) == {"k_sem", "k_geo", "v"}
    assert got["k_sem"].shape == (1, 3, 4)
    assert got["k_geo"].shape == (1, 3, 6)
    assert got["v"].shape == (1, 3, 8)


def test_decoupled_layer_kv_cache_get_slice_many() -> None:
    """Test the generic get_slice_many API."""
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    cache = DecoupledLayerKVCache(
        batch_size=1,
        max_seq_len=8,
        k_sem_dim=4,
        k_geo_dim=6,
        v_dim=8,
        k_sem_cfg=cfg,
        k_geo_cfg=cfg,
        v_cfg=cfg,
        device=torch.device("cpu"),
    )

    k_sem = torch.randn(1, 5, 4, dtype=torch.float32)
    k_geo = torch.randn(1, 5, 6, dtype=torch.float32)
    v = torch.randn(1, 5, 8, dtype=torch.float32)
    cache.append_many({"k_sem": k_sem, "k_geo": k_geo, "v": v})

    got = cache.get_slice_many(1, 4, dtype=torch.float32)
    assert set(got.keys()) == {"k_sem", "k_geo", "v"}
    assert got["k_sem"].shape == (1, 3, 4)
    assert got["k_geo"].shape == (1, 3, 6)
    assert got["v"].shape == (1, 3, 8)