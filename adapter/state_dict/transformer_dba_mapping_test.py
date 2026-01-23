"""
transformer_dba_mapping_test validates the adapter mapping + DBA surgery math.

This ensures that:
- V/O copying is exact (when shapes match)
- Q/K SVD initialization writes the expected compressed rows
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from typing import Any, cast

import torch
from torch import Tensor

# Allow running under an installed `caramba` package or directly from a repo checkout.
try:
    from caramba.adapter.state_dict import AdapterStateDictTransformer  # type: ignore[import-not-found]
    from caramba.config.model import ModelConfig  # type: ignore[import-not-found]
    from caramba.model import Model  # type: ignore[import-not-found]
except ModuleNotFoundError:
    # File lives at: <repo_root>/adapter/state_dict/transformer_dba_mapping_test.py
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from adapter.state_dict import AdapterStateDictTransformer
    from config.model import ModelConfig
    from model import Model

try:
    from caramba.initializers.dba.svd import DBASVD  # type: ignore[import-not-found]
except ModuleNotFoundError:
    from initializers.dba.svd import DBASVD


def _small_model_payload(*, mode: str) -> dict[str, Any]:
    d_model = 32
    n_heads = 4
    n_kv_heads = 2
    n_layers = 2
    d_ff = 64
    vocab_size = 128

    attn_node: dict[str, Any] = {
        "type": "AttentionLayer",
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "mode": str(mode),
        "rope_enabled": False,
        "is_causal": True,
        "dropout_p": 0.0,
        "bias": False,
    }
    if mode == "decoupled":
        attn_node.update(
            {
                "attn_dim": d_model,  # V dim
                "sem_dim": 8,
                "geo_dim": 8,
                "decoupled_gate": True,
            }
        )

    return {
        "type": "TransformerModel",
        "tied_embeddings": False,
        "embedder": {"type": "token", "vocab_size": vocab_size, "d_model": d_model},
        "topology": {
            "type": "StackedTopology",
            "layers": [
                {
                    "type": "NestedTopology",
                    "repeat": n_layers,
                    "layers": [
                        {
                            "type": "ResidualTopology",
                            "layers": [
                                {"type": "RMSNormLayer", "d_model": d_model, "eps": 1e-5},
                                attn_node,
                            ],
                        },
                        {
                            "type": "ResidualTopology",
                            "layers": [
                                {"type": "RMSNormLayer", "d_model": d_model, "eps": 1e-5},
                                {"type": "SwiGLULayer", "d_model": d_model, "d_ff": d_ff, "bias": False},
                            ],
                        },
                    ],
                },
                {"type": "RMSNormLayer", "d_model": d_model, "eps": 1e-5},
                {"type": "LinearLayer", "d_in": d_model, "d_out": vocab_size, "bias": False},
            ],
        },
    }


def _export_llama_like_state_dict(*, model: torch.nn.Module, adapter: AdapterStateDictTransformer) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}

    embedder = adapter.findEmbedder(model=model)
    head = adapter.findHead(model=model)
    attn = adapter.findAttn(model=model)
    mlp = adapter.findMlp(model=model)
    norms = adapter.findNorms(model=model)
    adapter.validateCounts(attn=attn, mlp=mlp, norms=norms)

    # Embeddings
    if embedder.token_embedding is None:
        raise RuntimeError("test model missing token_embedding")
    state[adapter.key(adapter.schema.embedder.tokens_weight)] = embedder.token_embedding.weight.detach().clone()

    for i, (a, m) in enumerate(zip(attn, mlp)):
        layer_prefix = adapter.layerPrefix(i=int(i))

        # Norms (2 per block)
        n0 = norms[2 * int(i)]
        n1 = norms[2 * int(i) + 1]
        if n0.weight is None or n1.weight is None:
            raise RuntimeError("test model missing RMSNorm weights")
        state[adapter.key(layer_prefix, adapter.schema.block.input_norm_weight)] = n0.weight.detach().clone()
        state[adapter.key(layer_prefix, adapter.schema.block.post_attn_norm_weight)] = n1.weight.detach().clone()

        # Attention (standard export)
        qkv = getattr(a, "qkv_proj", None)
        out = getattr(a, "out_proj", None)
        if qkv is None or out is None:
            raise RuntimeError("expected standard attention (qkv_proj/out_proj)")
        wqkv = cast(torch.nn.Linear, qkv).weight.detach()
        wo = cast(torch.nn.Linear, out).weight.detach()
        q_dim = int(wo.size(1))
        kv_dim = (int(wqkv.size(0)) - q_dim) // 2
        wq, wk, wv = wqkv.split([q_dim, kv_dim, kv_dim], dim=0)

        attn_prefix = adapter.key(layer_prefix, adapter.schema.attention.path)
        state[adapter.key(attn_prefix, adapter.schema.attention.q_weight)] = wq.contiguous().clone()
        state[adapter.key(attn_prefix, adapter.schema.attention.k_weight)] = wk.contiguous().clone()
        state[adapter.key(attn_prefix, adapter.schema.attention.v_weight)] = wv.contiguous().clone()
        state[adapter.key(attn_prefix, adapter.schema.attention.o_weight)] = wo.contiguous().clone()

        # MLP
        w_gate_up = cast(torch.nn.Linear, m.w_gate_up).weight.detach()
        w_gate, w_up = w_gate_up.chunk(2, dim=0)
        w_down = cast(torch.nn.Linear, m.w_down).weight.detach()
        mlp_prefix = adapter.key(layer_prefix, adapter.schema.mlp.path)
        state[adapter.key(mlp_prefix, adapter.schema.mlp.gate_weight)] = w_gate.contiguous().clone()
        state[adapter.key(mlp_prefix, adapter.schema.mlp.up_weight)] = w_up.contiguous().clone()
        state[adapter.key(mlp_prefix, adapter.schema.mlp.down_weight)] = w_down.contiguous().clone()

    # Final norm
    final_norm = norms[-1]
    if final_norm.weight is None:
        raise RuntimeError("test model missing final RMSNorm weight")
    state[adapter.key(adapter.schema.final_norm_weight)] = final_norm.weight.detach().clone()

    # Head
    if head is not None:
        state[adapter.key(adapter.schema.head.weight)] = head.linear.weight.detach().clone()

    return state


class TransformerDBAMappingTest(unittest.TestCase):
    def test_vo_copy_and_qk_svd_rows(self) -> None:
        torch.manual_seed(0)

        base_cfg = ModelConfig.model_validate(_small_model_payload(mode="standard"))
        baseline = Model(base_cfg).eval()

        adapter = AdapterStateDictTransformer.llama(dba_init="svd")
        teacher_state = _export_llama_like_state_dict(model=baseline, adapter=adapter)

        # Build student (decoupled) and apply surgery init.
        dba_cfg = ModelConfig.model_validate(_small_model_payload(mode="decoupled"))
        student = Model(dba_cfg).eval()
        adapter.apply(model=student, state_dict=teacher_state)

        # Grab one attention layer from student to inspect weights.
        attn_layers = adapter.findAttn(model=student)
        self.assertTrue(attn_layers)
        attn0 = attn_layers[0]

        # Extract teacher Q/K/V/O tensors for layer 0 from external dict.
        layer_prefix = adapter.layerPrefix(i=0)
        attn_prefix = adapter.key(layer_prefix, adapter.schema.attention.path)
        q = teacher_state[adapter.key(attn_prefix, adapter.schema.attention.q_weight)]
        k = teacher_state[adapter.key(attn_prefix, adapter.schema.attention.k_weight)]
        v = teacher_state[adapter.key(attn_prefix, adapter.schema.attention.v_weight)]
        o = teacher_state[adapter.key(attn_prefix, adapter.schema.attention.o_weight)]

        # V/O should be copied exactly (shapes match in this tiny config).
        v_proj = cast(torch.nn.Linear, getattr(attn0, "v_proj"))
        o_proj = cast(torch.nn.Linear, getattr(attn0, "out_proj"))
        self.assertTrue(torch.allclose(v_proj.weight.detach(), v, atol=0, rtol=0))
        # O may be truncated if o_in differs; in this config it matches.
        self.assertTrue(torch.allclose(o_proj.weight.detach(), o, atol=0, rtol=0))

        # Q/K should match the DBASVD compressed rows written into sem/geo weights.
        init = DBASVD()

        q_sem = cast(torch.nn.Linear, getattr(attn0, "q_sem")).weight.detach()
        q_geo = cast(torch.nn.Linear, getattr(attn0, "q_geo")).weight.detach()
        sem_q_dim = int(q_sem.size(0))
        geo_q_dim = int(q_geo.size(0))
        rank_q = sem_q_dim + geo_q_dim
        _Uq, Sq, Vhq = torch.linalg.svd(q.to(torch.float32), full_matrices=False)
        Wrq = (Sq[:rank_q].view(-1, 1) * Vhq[:rank_q, :]).contiguous()
        self.assertTrue(torch.allclose(q_sem.to(torch.float32), Wrq[:sem_q_dim, :], atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(q_geo.to(torch.float32), Wrq[sem_q_dim : sem_q_dim + geo_q_dim, :], atol=1e-4, rtol=1e-4))

        k_sem = cast(torch.nn.Linear, getattr(attn0, "k_sem")).weight.detach()
        k_geo = cast(torch.nn.Linear, getattr(attn0, "k_geo")).weight.detach()
        sem_k_dim = int(k_sem.size(0))
        geo_k_dim = int(k_geo.size(0))
        rank_k = sem_k_dim + geo_k_dim
        _Uk, Sk, Vhk = torch.linalg.svd(k.to(torch.float32), full_matrices=False)
        Wrk = (Sk[:rank_k].view(-1, 1) * Vhk[:rank_k, :]).contiguous()
        self.assertTrue(torch.allclose(k_sem.to(torch.float32), Wrk[:sem_k_dim, :], atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(k_geo.to(torch.float32), Wrk[sem_k_dim : sem_k_dim + geo_k_dim, :], atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()

