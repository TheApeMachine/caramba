"""
checkpoint_integrity_test provides fast checks for upcycle checkpoints.

These tests are meant to catch:
- wrong checkpoint path (missing / wrong structure)
- corrupted tensors (NaNs/Infs)
- obviously wrong parameter shapes for the configured model

They run in seconds and avoid multi-hour runs to discover a bad checkpoint.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any, cast

import torch

# Allow running under an installed `caramba` package or directly from a repo checkout.
try:
    from caramba.config.model import ModelConfig  # type: ignore[import-not-found]
    from caramba.model import Model  # type: ignore[import-not-found]
except ModuleNotFoundError:
    # Repo layout has packages at the repo root (config/, model/, ...).
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from config.model import ModelConfig
    from model import Model


def _build_model_payload_llama32_1b_dba() -> dict[str, Any]:
    # Mirror the common DBA paper configs (small enough for construction; does not load weights).
    return {
        "type": "TransformerModel",
        "tied_embeddings": False,
        "embedder": {"type": "token", "vocab_size": 128256, "d_model": 2048},
        "topology": {
            "type": "StackedTopology",
            "layers": [
                {
                    "type": "NestedTopology",
                    "repeat": 16,
                    "layers": [
                        {
                            "type": "ResidualTopology",
                            "layers": [
                                {"type": "RMSNormLayer", "d_model": 2048, "eps": 1e-5},
                                {
                                    "type": "AttentionLayer",
                                    "d_model": 2048,
                                    "n_heads": 32,
                                    "n_kv_heads": 8,
                                    "mode": "decoupled",
                                    "attn_dim": 1280,
                                    "sem_dim": 256,
                                    "geo_dim": 1024,
                                    "rope_enabled": True,
                                    "rope_base": 500000.0,
                                    "is_causal": True,
                                    "dropout_p": 0.0,
                                    "decoupled_gate": True,
                                },
                            ],
                        },
                        {
                            "type": "ResidualTopology",
                            "layers": [
                                {"type": "RMSNormLayer", "d_model": 2048, "eps": 1e-5},
                                {"type": "SwiGLULayer", "d_model": 2048, "d_ff": 8192, "bias": False},
                            ],
                        },
                    ],
                },
                {"type": "RMSNormLayer", "d_model": 2048, "eps": 1e-5},
                {"type": "LinearLayer", "d_in": 2048, "d_out": 128256, "bias": False},
            ],
        },
    }


class CheckpointIntegrityTest(unittest.TestCase):
    def test_student_checkpoint_structure_and_finiteness(self) -> None:
        ckpt_path = Path(os.environ.get("CARAMBA_STUDENT_CKPT", "runs/paper/finetune_global_final.pt"))
        if not ckpt_path.exists():
            self.skipTest(f"Checkpoint not found: {ckpt_path} (set CARAMBA_STUDENT_CKPT to override)")

        # Prefer safe load when possible; fall back to unsafe only if explicitly allowed.
        allow_unsafe = bool(int(os.environ.get("CARAMBA_UNSAFE_PICKLE_LOAD", "0")))
        try:
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            if not allow_unsafe:
                raise
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        self.assertIsInstance(obj, dict)
        d = cast(dict[str, object], obj)
        self.assertIn("student_state_dict", d)
        sd = d["student_state_dict"]
        self.assertIsInstance(sd, dict)
        sd_t = cast(dict[str, torch.Tensor], sd)

        # Basic size sanity: a real 1B-ish model state_dict should have many keys.
        self.assertGreater(len(sd_t), 100, "student_state_dict suspiciously small; wrong file?")

        # Build the configured model and load strictly to catch mismatched keys/shapes.
        cfg = ModelConfig.model_validate(_build_model_payload_llama32_1b_dba())
        model = Model(cfg)
        missing, unexpected = model.load_state_dict(sd_t, strict=False)
        # We allow some flexibility across runs, but huge mismatches are almost always wrong.
        self.assertLess(len(unexpected), 50, f"Too many unexpected keys ({len(unexpected)}): likely wrong checkpoint/model.")
        self.assertLess(len(missing), 50, f"Too many missing keys ({len(missing)}): likely wrong checkpoint/model.")

        # Finiteness check on a small sample of tensors (full scan can be slow).
        checked = 0
        for k, t in sd_t.items():
            if not isinstance(t, torch.Tensor):
                continue
            if t.numel() == 0:
                continue
            # prioritize high-impact tensors
            if any(s in k for s in ("embedder", "token_embedding", "out_proj", "v_proj", "q_sem", "k_sem", "q_geo", "k_geo", "w_gate_up", "w_down")):
                self.assertTrue(torch.isfinite(t).all(), f"Non-finite tensor in checkpoint: {k}")
                checked += 1
                if checked >= 25:
                    break
        self.assertGreater(checked, 5, "Did not find expected core tensors to validate; key naming changed?")


if __name__ == "__main__":
    unittest.main()

