"""
transformer_dba_smoke_test provides fast sanity checks for DBA surgery + distillation.

Goal:
- Catch catastrophic DBA-surgery regressions in seconds (CPU-only), instead of
  discovering them via multi-hour runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from typing import Any, cast

import torch
from torch import Tensor

# Allow running as:
# - `python -m caramba.adapter.state_dict.transformer_dba_smoke_test` (preferred)
# - `python adapter/state_dict/transformer_dba_smoke_test.py` (fallback)
try:
    from caramba.adapter.state_dict import AdapterStateDictTransformer
    from caramba.config.model import ModelConfig
    from caramba.model import Model
    from caramba.trainer.blockwise import BlockwiseConfig, BlockwiseTrainer
    from caramba.trainer.distill import DistillLoss
except ModuleNotFoundError:
    # When executed as a script, Python may not have the repo root on sys.path.
    # Add the directory containing the `caramba/` package.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from caramba.adapter.state_dict import AdapterStateDictTransformer
    from caramba.config.model import ModelConfig
    from caramba.model import Model
    from caramba.trainer.blockwise import BlockwiseConfig, BlockwiseTrainer
    from caramba.trainer.distill import DistillLoss


def _logits(model: torch.nn.Module, x: Tensor) -> Tensor:
    out = model(x)
    if isinstance(out, tuple) and out:
        out = out[0]
    if hasattr(out, "logits"):
        out = out.logits  # type: ignore[assignment]
    return cast(Tensor, out)


def _small_model_payload(*, mode: str) -> dict[str, Any]:
    """Return a tiny Llama-like model payload (manifest-style dict)."""
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
        # Keep QK smaller than V (tests the pad-to-v_head_dim logic too).
        attn_node.update(
            {
                "attn_dim": d_model,  # V dim
                "sem_dim": 8,  # 2 per head
                "geo_dim": 8,  # 2 per head
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
    """Build an external (HF-style) llama state_dict from a *standard* Caramba model.

    This lets us test the full adapter+surgery path without downloading HF weights.
    """
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

    # Blocks
    for i, (a, m) in enumerate(zip(attn, mlp)):
        layer_prefix = adapter.layerPrefix(i=int(i))

        # Norms (2 per block)
        n0 = norms[2 * int(i)]
        n1 = norms[2 * int(i) + 1]
        if n0.weight is None or n1.weight is None:
            raise RuntimeError("test model missing RMSNorm weights")
        state[adapter.key(layer_prefix, adapter.schema.block.input_norm_weight)] = n0.weight.detach().clone()
        state[adapter.key(layer_prefix, adapter.schema.block.post_attn_norm_weight)] = n1.weight.detach().clone()

        # Attention (standard only for export)
        qkv = getattr(a, "qkv_proj", None)
        out = getattr(a, "out_proj", None)
        if qkv is None or out is None:
            raise RuntimeError("expected standard attention (qkv_proj/out_proj)")
        wqkv = cast(torch.nn.Linear, qkv).weight.detach()
        wo = cast(torch.nn.Linear, out).weight.detach()

        q_dim = int(getattr(a, "_q_dim", 0)) or int(wqkv.size(0)) - 2 * int(wqkv.size(0) // 4)  # fallback
        # Better: derive from config-like split (q, k, v) used by StandardAttentionLayer.
        # In this tiny model, q_dim == d_model and kv_dim == n_kv_heads * head_dim.
        # We can infer kv_dim from wo.shape (o is [d_model, q_dim]).
        q_dim = int(wo.size(1))
        kv_dim = (int(wqkv.size(0)) - q_dim) // 2

        wq, wk, wv = wqkv.split([q_dim, kv_dim, kv_dim], dim=0)
        attn_prefix = adapter.key(layer_prefix, adapter.schema.attention.path)
        state[adapter.key(attn_prefix, adapter.schema.attention.q_weight)] = wq.contiguous().clone()
        state[adapter.key(attn_prefix, adapter.schema.attention.k_weight)] = wk.contiguous().clone()
        state[adapter.key(attn_prefix, adapter.schema.attention.v_weight)] = wv.contiguous().clone()
        state[adapter.key(attn_prefix, adapter.schema.attention.o_weight)] = wo.contiguous().clone()

        # MLP (SwiGLU): split fused gate+up
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

    # Head (explicit)
    if head is not None:
        state[adapter.key(adapter.schema.head.weight)] = head.linear.weight.detach().clone()

    return state


class TransformerDBASmokeTest(unittest.TestCase):
    def test_surgery_init_is_finite_and_not_explosive(self) -> None:
        torch.manual_seed(0)

        # Build a tiny baseline (standard attention) model and export a llama-like state_dict.
        base_payload = _small_model_payload(mode="standard")
        base_cfg = ModelConfig.model_validate(base_payload)
        baseline = Model(base_cfg).eval()

        adapter = AdapterStateDictTransformer.llama(dba_init="svd")
        teacher_state = _export_llama_like_state_dict(model=baseline, adapter=adapter)

        # Round-trip: loading the exported state_dict into a fresh baseline should match outputs.
        baseline2 = Model(base_cfg).eval()
        adapter.apply(model=baseline2, state_dict=teacher_state)

        B, T = 2, 8
        vocab = int(base_payload["embedder"]["vocab_size"])
        x = torch.randint(0, vocab, (B, T), dtype=torch.long)

        y0 = _logits(baseline, x)
        y1 = _logits(baseline2, x)
        self.assertTrue(torch.allclose(y0, y1, atol=1e-5, rtol=1e-5))

        # DBA student: apply surgery init from the *same* teacher_state.
        dba_payload = _small_model_payload(mode="decoupled")
        dba_cfg = ModelConfig.model_validate(dba_payload)
        student = Model(dba_cfg).eval()
        adapter.apply(model=student, state_dict=teacher_state)

        ys = _logits(student, x)
        self.assertTrue(torch.isfinite(ys).all())
        # Guard against scale explosions that would make CE/perplexity meaningless.
        self.assertLess(float(ys.detach().abs().max().item()), 1.0e4)

    def test_blockwise_distillation_reduces_mismatch(self) -> None:
        torch.manual_seed(0)

        # Teacher baseline from exported llama-like weights.
        base_payload = _small_model_payload(mode="standard")
        base_cfg = ModelConfig.model_validate(base_payload)
        baseline = Model(base_cfg).eval()
        adapter = AdapterStateDictTransformer.llama(dba_init="svd")
        teacher_state = _export_llama_like_state_dict(model=baseline, adapter=adapter)
        teacher = Model(base_cfg).eval()
        adapter.apply(model=teacher, state_dict=teacher_state)

        # Student starts from DBA surgery init.
        dba_payload = _small_model_payload(mode="decoupled")
        dba_cfg = ModelConfig.model_validate(dba_payload)
        student = Model(dba_cfg)
        adapter.apply(model=student, state_dict=teacher_state)

        # Blockwise distillation over attention modules only.
        opt = torch.optim.SGD(student.parameters(), lr=0.5)
        trainer = BlockwiseTrainer(
            teacher=teacher,
            student=student,
            optimizer=opt,
            loss=DistillLoss(),
            predicate=lambda _name, m: type(m).__name__ in {"AttentionLayer", "StandardAttentionLayer", "DecoupledAttentionLayer"},
            config=BlockwiseConfig(cache_teacher_outputs=False, use_truncated_forward=True),
        )

        vocab = int(base_payload["embedder"]["vocab_size"])
        x = torch.randint(0, vocab, (2, 8), dtype=torch.long)

        # Train only the first attention block a few times on the same batch.
        l0 = float(trainer.step(x, block_index=0).item())
        for _ in range(10):
            _ = trainer.step(x, block_index=0)
        l1 = float(trainer.step(x, block_index=0).item())

        self.assertLess(l1, l0)


if __name__ == "__main__":
    unittest.main()

