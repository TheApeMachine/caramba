#!/usr/bin/env python3
"""Verify that surgery properly loads FFN weights from teacher.

This script checks that:
1. Teacher model (Llama) produces sensible output
2. Surgery correctly copies FFN/embedding weights to DBA model
3. The DBA model has the same FFN weights as teacher after surgery
"""

import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parent.parent))

from caramba.layer.mlx.standard_attention import TeacherModel


def main():
    # Find weights
    cache_dir = Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-1B"
    weights_file = None
    for snapshot_dir in cache_dir.glob("snapshots/*"):
        wf = snapshot_dir / "model.safetensors"
        if wf.exists():
            weights_file = wf
            break

    if weights_file is None:
        print("ERROR: Could not find Llama weights in cache")
        return

    print(f"Loading weights from: {weights_file}")

    # Load raw weights
    raw_weights = mx.load(str(weights_file))
    print(f"\n=== Raw Llama weight keys (first 20) ===")
    for i, key in enumerate(sorted(raw_weights.keys())[:20]):
        print(f"  {key}: {raw_weights[key].shape}")

    # Create teacher model
    teacher = TeacherModel(
        d_model=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        d_ff=8192,
        vocab_size=128256,
        rope_base=500000.0,
    )

    # Print model parameter names
    print(f"\n=== Teacher model parameter names (first 20) ===")
    params = tree_flatten(teacher.parameters())
    for i, (name, arr) in enumerate(params[:20]):
        print(f"  {name}: {arr.shape}")

    # Now we can see what mapping is needed!
    print("\n=== Required mapping ===")
    print("Llama key -> Teacher key")
    print("model.embed_tokens.weight -> embed_tokens.weight")
    print("model.layers.0.input_layernorm.weight -> layers.0.norm1.weight")
    print("model.layers.0.self_attn.q_proj.weight -> layers.0.attention.q_proj.weight")
    print("model.layers.0.mlp.gate_proj.weight -> layers.0.w_gate.weight")


def main_original():
    # Find weights
    cache_dir = Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-1B"
    weights_file = None
    for snapshot_dir in cache_dir.glob("snapshots/*"):
        wf = snapshot_dir / "model.safetensors"
        if wf.exists():
            weights_file = wf
            break

    if weights_file is None:
        print("ERROR: Could not find Llama weights in cache")
        return

    print(f"Loading weights from: {weights_file}")

    # Load raw weights
    raw_weights = mx.load(str(weights_file))
    print(f"\n=== Raw weights file has {len(raw_weights)} keys ===")

    # Step 1: Verify teacher model
    print("\n" + "=" * 60)
    print("Step 1: Verify Teacher Model (Llama)")
    print("=" * 60)

    teacher = load_teacher_from_llama(weights_file)

    # Check embed tokens
    teacher_embed_sum = float(mx.sum(teacher.embed_tokens.weight))
    raw_embed_sum = float(mx.sum(raw_weights["model.embed_tokens.weight"]))
    print(f"Teacher embed_tokens sum: {teacher_embed_sum:.2f}")
    print(f"Raw weights embed sum:    {raw_embed_sum:.2f}")
    print(f"Match: {abs(teacher_embed_sum - raw_embed_sum) < 1.0}")

    # Check FFN weights
    teacher_gate_sum = float(mx.sum(teacher.layers[0].ffn.w_gate.weight))
    raw_gate_sum = float(mx.sum(raw_weights["model.layers.0.mlp.gate_proj.weight"]))
    print(f"Teacher layer0 FFN gate sum: {teacher_gate_sum:.2f}")
    print(f"Raw weights gate sum:        {raw_gate_sum:.2f}")
    print(f"Match: {abs(teacher_gate_sum - raw_gate_sum) < 1.0}")

    # Quick generation test
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    input_ids = tokenizer.encode("The quick brown fox", add_special_tokens=False)
    tokens = mx.array([input_ids])

    logits, _, _ = teacher(tokens)
    print(f"\nTeacher output logits shape: {logits.shape}")

    # Get top prediction
    last_logits = logits[0, -1, :]
    top_idx = int(mx.argmax(last_logits))
    print(f"Teacher predicts next token: {repr(tokenizer.decode([top_idx]))}")

    # Step 2: Test Surgery
    print("\n" + "=" * 60)
    print("Step 2: Test Surgery (DBA with Llama FFN)")
    print("=" * 60)

    surgery = AttentionSurgeryMLX(
        sem_dim=256,
        geo_dim=512,
        v_dim=768,
    )

    # Create DBA model
    model = surgery.create_dba_model()
    print(f"Created DBA model with {len(model.layers)} layers")

    # Check weights BEFORE surgery
    pre_embed_sum = float(mx.sum(model.embed_tokens.weight))
    pre_gate_sum = float(mx.sum(model.layers[0].ffn.w_gate.weight))
    print(f"\nBEFORE surgery:")
    print(f"  DBA embed_tokens sum: {pre_embed_sum:.2f}")
    print(f"  DBA layer0 FFN gate sum: {pre_gate_sum:.2f}")

    # Apply surgery
    model = surgery.apply_surgery(model, raw_weights, init_mode="fresh")

    # Check weights AFTER surgery
    post_embed_sum = float(mx.sum(model.embed_tokens.weight))
    post_gate_sum = float(mx.sum(model.layers[0].ffn.w_gate.weight))
    print(f"\nAFTER surgery:")
    print(f"  DBA embed_tokens sum: {post_embed_sum:.2f} (should match raw: {raw_embed_sum:.2f})")
    print(f"  DBA layer0 FFN gate sum: {post_gate_sum:.2f} (should match raw: {raw_gate_sum:.2f})")
    print(f"  Embed match: {abs(post_embed_sum - raw_embed_sum) < 1.0}")
    print(f"  FFN match: {abs(post_gate_sum - raw_gate_sum) < 1.0}")

    # More detailed FFN check
    print("\n--- Detailed FFN weight check ---")
    for i in [0, 7, 15]:
        dba_gate = model.layers[i].ffn.w_gate.weight
        raw_gate = raw_weights[f"model.layers.{i}.mlp.gate_proj.weight"]

        dba_sum = float(mx.sum(dba_gate))
        raw_sum = float(mx.sum(raw_gate))
        match = abs(dba_sum - raw_sum) < 1.0

        print(f"Layer {i} FFN gate: DBA={dba_sum:.2f}, Raw={raw_sum:.2f}, Match={match}")

    # Test DBA generation (with random attention, should not be coherent)
    print("\n--- DBA model forward pass ---")
    logits, _, _ = model(tokens)
    print(f"DBA output logits shape: {logits.shape}")

    last_logits = logits[0, -1, :]
    print(f"Logits min/max/mean: {float(last_logits.min()):.2f} / {float(last_logits.max()):.2f} / {float(mx.mean(last_logits)):.2f}")

    # Top predictions (won't be coherent with random attention, but should work)
    top_idx = int(mx.argmax(last_logits))
    print(f"DBA predicts next token: {repr(tokenizer.decode([top_idx]))}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if abs(post_embed_sum - raw_embed_sum) < 1.0 and abs(post_gate_sum - raw_gate_sum) < 1.0:
        print("SUCCESS: Surgery correctly copied FFN and embedding weights!")
    else:
        print("FAILURE: Weights were not copied correctly")
        print(f"  Expected embed sum: {raw_embed_sum:.2f}, got: {post_embed_sum:.2f}")
        print(f"  Expected gate sum: {raw_gate_sum:.2f}, got: {post_gate_sum:.2f}")


if __name__ == "__main__":
    main()
