from __future__ import annotations

import pytest
import torch

from caramba.cache.tensor import SeqCacheTensor
from caramba.config.kvcache import KVCacheKind, KVCacheTensorConfig


def test_seq_cache_tensor_fp16_append_get_and_truncate() -> None:
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    c = SeqCacheTensor(batch_size=2, max_seq_len=8, dim=4, cfg=cfg, device=torch.device("cpu"))

    x0 = torch.randn(2, 3, 4, dtype=torch.float32)
    old = c.append(x0)
    assert old == 0
    assert c.pos == 3
    got = c.get(dtype=torch.float32)
    assert got.shape == (2, 3, 4)
    # Stored as fp16 internally; compare against the fp16-rounded representation.
    expect = x0.to(torch.float16).to(torch.float32)
    assert torch.allclose(got, expect, atol=0, rtol=0)

    # Truncate back and ensure get reflects new length.
    c.truncate(1)
    assert c.pos == 1
    got2 = c.get(dtype=torch.float32)
    assert got2.shape == (2, 1, 4)
    assert torch.allclose(got2, expect[:, :1], atol=0, rtol=0)


def test_seq_cache_tensor_validates_append_and_slice_bounds() -> None:
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    c = SeqCacheTensor(batch_size=1, max_seq_len=4, dim=4, cfg=cfg, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="dim mismatch"):
        c.append(torch.randn(1, 1, 5))

    c.append(torch.randn(1, 4, 4))
    with pytest.raises(ValueError, match="Cache overflow"):
        c.append(torch.randn(1, 1, 4))

    with pytest.raises(ValueError, match="Invalid slice"):
        c.get_slice(2, 1)
    with pytest.raises(ValueError, match="Requested end"):
        c.get_slice(0, 5)


def test_seq_cache_tensor_empty_slice_has_correct_shape() -> None:
    cfg = KVCacheTensorConfig(kind=KVCacheKind.FP16, qblock=32, residual_len=0)
    c = SeqCacheTensor(batch_size=3, max_seq_len=8, dim=4, cfg=cfg, device=torch.device("cpu"))
    out = c.get_slice(0, 0, dtype=torch.float32)
    assert out.shape == (3, 0, 4)


def test_seq_cache_tensor_q8_residual_window_and_concat_path() -> None:
    # Residual buffer should serve tail slices directly, and mixed slices should
    # concatenate quantized-prefix + residual-tail.
    cfg = KVCacheTensorConfig(kind=KVCacheKind.Q8_0, qblock=4, residual_len=4)
    c = SeqCacheTensor(batch_size=1, max_seq_len=16, dim=8, cfg=cfg, device=torch.device("cpu"))
    assert c.is_quantized is True

    x = torch.randn(1, 10, 8, dtype=torch.float16)
    c.append(x)
    assert c.pos == 10

    # Tail slice entirely within residual window.
    tail = c.get_slice(6, 10, dtype=torch.float16)
    assert tail.shape == (1, 4, 8)

    # Mixed slice that spans prefix (quantized) and tail (residual).
    mixed = c.get_slice(2, 10, dtype=torch.float16)
    assert mixed.shape == (1, 8, 8)

    # The values should be close-ish to original after quantize/dequantize;
    # but the residual portion should be very close since it comes from fp16 tail.
    assert torch.allclose(mixed[:, -4:], x[:, -4:], atol=1e-3, rtol=1e-3)


def test_seq_cache_tensor_truncate_rebuilds_residual() -> None:
    cfg = KVCacheTensorConfig(kind=KVCacheKind.Q8_0, qblock=4, residual_len=4)
    c = SeqCacheTensor(batch_size=1, max_seq_len=16, dim=8, cfg=cfg, device=torch.device("cpu"))
    x = torch.randn(1, 8, 8, dtype=torch.float16)
    c.append(x)

    c.truncate(5)
    assert c.pos == 5
    # Tail slice should be reconstructible and have correct shape.
    tail = c.get_slice(1, 5, dtype=torch.float16)
    assert tail.shape == (1, 4, 8)

    with pytest.raises(ValueError, match="Invalid truncate"):
        c.truncate(6)

