from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from carmath.batching import token_budget_batch_size
from carmath.bytes import bytes_per_kind
from carmath.linalg import randomized_svd
from carmath.optim import global_grad_norm_l2
from carmath.precision import (
    autocast_dtype,
    autocast_dtype_str,
    weight_dtype,
    weight_dtype_str,
)
from carmath.sketch import sketch_dot5, stable_int_hash, stride_sketch_indices
from carmath.splits import train_val_counts


def test_token_budget_batch_size_scales_and_clamps() -> None:
    # Basic scale: when block size doubles, batch size halves.
    assert token_budget_batch_size(64, block_size=1024, ref_block_size=512) == 32

    # min clamp
    assert token_budget_batch_size(1, block_size=4096, ref_block_size=512, min_batch_size=4) == 4

    # invalid sizes fall back to base batch size (with min clamp)
    assert token_budget_batch_size(3, block_size=0, ref_block_size=512, min_batch_size=1) == 3
    assert token_budget_batch_size(3, block_size=512, ref_block_size=0, min_batch_size=5) == 5

    # base batch size <= 0 becomes 1
    assert token_budget_batch_size(0, block_size=512, ref_block_size=512, min_batch_size=1) == 1


def test_train_val_counts_edge_cases() -> None:
    assert train_val_counts(0, 0.1) == (0, 0)
    assert train_val_counts(10, 0.0) == (10, 0)
    assert train_val_counts(10, -1.0) == (10, 0)

    # Ensure at least 1 train sample when possible
    n_train, n_val = train_val_counts(2, 0.9)
    assert (n_train, n_val) == (1, 1)

    # min_val enforced when val_frac > 0
    n_train, n_val = train_val_counts(100, 0.001, min_val=7)
    assert n_val == 7
    assert n_train == 93


def test_bytes_per_kind_known_and_unknown() -> None:
    assert bytes_per_kind("fp32") == 4.0
    assert bytes_per_kind("FP16") == 2.0
    assert bytes_per_kind("q8_0") == 1.0
    assert bytes_per_kind("q4_0") == 0.625
    with pytest.raises(ValueError, match=r"Unknown kind: int8"):
        bytes_per_kind("int8")


def test_precision_helpers_cpu_defaults() -> None:
    cpu = torch.device("cpu")
    assert autocast_dtype_str(cpu) == "bfloat16"
    assert autocast_dtype(cpu, "auto") == torch.bfloat16
    assert autocast_dtype(cpu, "bfloat16") == torch.bfloat16
    assert autocast_dtype(cpu, "float16") == torch.float16

    assert weight_dtype_str(cpu) == "float32"
    assert weight_dtype(cpu, "auto") == torch.float32
    assert weight_dtype(cpu, "bfloat16") == torch.bfloat16
    assert weight_dtype(cpu, "float16") == torch.float16


def test_precision_helpers_mps_and_cuda_specs_are_handled() -> None:
    # These are purely type-based decisions; device availability should not crash.
    mps = torch.device("mps")
    assert autocast_dtype_str(mps) == "float16"
    assert weight_dtype_str(mps) == "float16"

    cuda = torch.device("cuda")
    assert autocast_dtype_str(cuda) in {"bfloat16", "float16"}
    assert weight_dtype_str(cuda) in {"bfloat16", "float16"}


def test_precision_helpers_cuda_bf16_supported_branch(monkeypatch) -> None:
    # Force the bf16 branch without requiring a working CUDA runtime.
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)
    cuda = torch.device("cuda")
    assert autocast_dtype_str(cuda) == "bfloat16"
    assert weight_dtype_str(cuda) == "bfloat16"
    assert autocast_dtype(cuda, "auto") == torch.bfloat16
    assert weight_dtype(cuda, "auto") == torch.bfloat16


def test_precision_helpers_cuda_bf16_not_supported_branch(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False, raising=False)
    cuda = torch.device("cuda")
    assert autocast_dtype_str(cuda) == "float16"
    assert weight_dtype_str(cuda) == "float16"
    assert autocast_dtype(cuda, "auto") == torch.float16
    assert weight_dtype(cuda, "auto") == torch.float16


def test_global_grad_norm_l2_matches_manual() -> None:
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(3, 4, bias=True), nn.Linear(4, 2, bias=False))
    x = torch.randn(5, 3)
    y = m(x).sum()
    y.backward()

    # Manual L2 norm over all grad elements.
    sq = 0.0
    for p in m.parameters():
        assert p.grad is not None
        sq += float(p.grad.detach().float().pow(2).sum())
    expected = math.sqrt(sq)
    got = global_grad_norm_l2(m)
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_global_grad_norm_l2_no_grads() -> None:
    m = nn.Linear(2, 2)
    # No backward call => grads are None
    assert global_grad_norm_l2(m) == 0.0


def test_randomized_svd_shapes_determinism_and_reconstruction() -> None:
    torch.manual_seed(0)
    A = torch.randn(20, 10)

    U1, S1, Vh1 = randomized_svd(A, rank=6, n_iter=1, oversample=4, seed="demo")
    U2, S2, Vh2 = randomized_svd(A, rank=6, n_iter=1, oversample=4, seed="demo")
    assert U1.shape == (20, 6)
    assert S1.shape == (6,)
    assert Vh1.shape == (6, 10)
    assert torch.allclose(U1, U2)
    assert torch.allclose(S1, S2)
    assert torch.allclose(Vh1, Vh2)

    # A simple sanity check: reconstruction should be better than "nothing".
    A_hat = U1 @ torch.diag(S1) @ Vh1
    err = torch.linalg.norm(A - A_hat) / torch.linalg.norm(A)
    assert float(err) < 0.75


def test_randomized_svd_validates_inputs() -> None:
    A = torch.randn(3, 3)
    with pytest.raises(ValueError, match=r"randomized_svd expects a 2D matrix"):
        randomized_svd(A.unsqueeze(0), rank=1)
    with pytest.raises(ValueError, match=r"Invalid shapes/rank for randomized_svd"):
        randomized_svd(A, rank=0)


def test_stable_int_hash_is_deterministic() -> None:
    assert stable_int_hash("abc") == stable_int_hash("abc")
    assert stable_int_hash("abc") != stable_int_hash("abcd")


def test_stride_sketch_indices_edge_cases_and_determinism() -> None:
    dev = torch.device("cpu")
    assert stride_sketch_indices(0, 10, seed="x", device=dev).numel() == 0
    assert stride_sketch_indices(10, 0, seed="x", device=dev).numel() == 0
    assert torch.equal(stride_sketch_indices(5, 10, seed="x", device=dev), torch.arange(5))

    idx1 = stride_sketch_indices(100, 7, seed="demo", device=dev, hashed_start=True)
    idx2 = stride_sketch_indices(100, 7, seed="demo", device=dev, hashed_start=True)
    assert torch.equal(idx1, idx2)
    assert idx1.min() >= 0
    assert idx1.max() < 100


def test_sketch_dot5_matches_torch_reference_with_and_without_idx() -> None:
    torch.manual_seed(0)
    w = torch.randn(4, 3)
    wp = w + 0.1 * torch.randn(4, 3)
    u = torch.randn(4, 3)
    g = torch.randn(4, 3)

    uu, tt, ut, vv, uv = sketch_dot5(w, wp, u, g, idx=None)
    # Torch reference:
    t = (w - wp).reshape(-1).float()
    u1 = u.reshape(-1).float()
    g1 = g.reshape(-1).float()
    assert uu == pytest.approx(float(torch.dot(u1, u1)), rel=1e-6, abs=1e-6)
    assert tt == pytest.approx(float(torch.dot(t, t)), rel=1e-6, abs=1e-6)
    assert ut == pytest.approx(float(torch.dot(u1, t)), rel=1e-6, abs=1e-6)
    assert vv == pytest.approx(float(torch.dot(g1, g1)), rel=1e-6, abs=1e-6)
    assert uv == pytest.approx(float(torch.dot(u1, g1)), rel=1e-6, abs=1e-6)

    # Indexed variant
    idx = torch.tensor([0, 2, 5, 7, 9], dtype=torch.long)
    uu2, tt2, ut2, vv2, uv2 = sketch_dot5(w, wp, u, g, idx=idx)
    wv = w.reshape(-1).index_select(0, idx).float()
    wpv = wp.reshape(-1).index_select(0, idx).float()
    uvv = u.reshape(-1).index_select(0, idx).float()
    gvv = g.reshape(-1).index_select(0, idx).float()
    t2 = wv - wpv
    assert uu2 == pytest.approx(float(torch.dot(uvv, uvv)), rel=1e-6, abs=1e-6)
    assert tt2 == pytest.approx(float(torch.dot(t2, t2)), rel=1e-6, abs=1e-6)
    assert ut2 == pytest.approx(float(torch.dot(uvv, t2)), rel=1e-6, abs=1e-6)
    assert vv2 == pytest.approx(float(torch.dot(gvv, gvv)), rel=1e-6, abs=1e-6)
    assert uv2 == pytest.approx(float(torch.dot(uvv, gvv)), rel=1e-6, abs=1e-6)


def test_sketch_dot5_with_no_grad_returns_zeros_for_grad_terms() -> None:
    torch.manual_seed(0)
    w = torch.randn(10)
    wp = w + 0.01 * torch.randn(10)
    u = torch.randn(10)

    uu, _tt, _ut, vv, uv = sketch_dot5(w, wp, u, g=None, idx=None)
    assert float(uu) > 0.0
    assert float(vv) == 0.0
    assert float(uv) == 0.0

