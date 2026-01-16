#!/usr/bin/env python3
"""Test Llama inference using the exact MLX example pattern."""

import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten


class LlamaAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads

        self.rope = nn.RoPE(dims // num_heads, traditional=True)
        self.query_proj = nn.Linear(dims, dims, bias=False)
        self.key_proj = nn.Linear(dims, dims, bias=False)
        self.value_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape

        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat), (keys, values)


class LlamaEncoderLayer(nn.Module):
    def __init__(self, dims: int, mlp_dims: int, num_heads: int):
        super().__init__()

        self.attention = LlamaAttention(dims, num_heads)

        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)

        self.linear1 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear2 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear3 = nn.Linear(mlp_dims, dims, bias=False)

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        y, cache = self.attention(y, y, y, mask, cache)
        x = x + y

        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        y = a * mx.sigmoid(a) * b
        y = self.linear3(y)
        x = x + y

        return x, cache


class Llama(nn.Module):
    def __init__(
        self, num_layers: int, vocab_size: int, dims: int, mlp_dims: int, num_heads: int
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.layers = [
            LlamaEncoderLayer(dims, mlp_dims, num_heads) for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(dims)
        self.out_proj = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        x = self.embedding(x)
        for l in self.layers:
            x, _ = l(x, mask)
        x = self.norm(x)
        return self.out_proj(x)

    def generate(self, x, temp=1.0):
        cache = []

        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        x = self.embedding(x)
        for l in self.layers:
            x, c = l(x, mask=mask)
            cache.append(c)
        x = self.norm(x)
        y = self.out_proj(x[:, -1])
        y = mx.random.categorical(y * (1/temp))

        yield y

        while True:
            x = y[:, None]

            x = self.embedding(x)
            for i in range(len(cache)):
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = self.out_proj(x[:, -1])
            y = mx.random.categorical(y * (1/temp))

            yield y


def convert_llama_weights(weights: dict) -> dict:
    """Convert HuggingFace Llama weights to MLX example format."""
    new_weights = {}

    for key, value in weights.items():
        new_key = key

        # Embedding
        if "model.embed_tokens.weight" in key:
            new_key = "embedding.weight"

        # Final norm
        elif "model.norm.weight" in key:
            new_key = "norm.weight"

        # LM head
        elif "lm_head.weight" in key:
            new_key = "out_proj.weight"

        # Layer weights
        elif "model.layers." in key:
            # Extract layer number
            parts = key.split(".")
            layer_idx = parts[2]

            # Norms
            if "input_layernorm" in key:
                new_key = f"layers.{layer_idx}.norm1.weight"
            elif "post_attention_layernorm" in key:
                new_key = f"layers.{layer_idx}.norm2.weight"

            # Attention
            elif "self_attn.q_proj" in key:
                new_key = f"layers.{layer_idx}.attention.query_proj.weight"
            elif "self_attn.k_proj" in key:
                new_key = f"layers.{layer_idx}.attention.key_proj.weight"
            elif "self_attn.v_proj" in key:
                new_key = f"layers.{layer_idx}.attention.value_proj.weight"
            elif "self_attn.o_proj" in key:
                new_key = f"layers.{layer_idx}.attention.out_proj.weight"

            # FFN
            elif "mlp.gate_proj" in key:
                new_key = f"layers.{layer_idx}.linear1.weight"
            elif "mlp.up_proj" in key:
                new_key = f"layers.{layer_idx}.linear2.weight"
            elif "mlp.down_proj" in key:
                new_key = f"layers.{layer_idx}.linear3.weight"
            else:
                continue
        else:
            continue

        new_weights[new_key] = value

    return new_weights


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
        print("ERROR: Could not find Llama weights")
        return

    print(f"Loading weights from: {weights_file}")

    # Load and convert weights
    raw_weights = mx.load(str(weights_file))
    converted = convert_llama_weights(raw_weights)

    print(f"Converted {len(converted)} weights")
    print("Sample keys:", list(converted.keys())[:5])

    # Create model - Llama 3.2 1B config
    model = Llama(
        num_layers=16,
        vocab_size=128256,
        dims=2048,
        mlp_dims=8192,
        num_heads=32,
    )

    # Load weights exactly like MLX example
    model.update(tree_unflatten(list(converted.items())))
    mx.eval(model.parameters())

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Generate
    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = mx.array([input_ids])

    print(f"\nPrompt: {prompt}")
    print("Generated: ", end="", flush=True)

    tokens = []
    for i, t in zip(range(50), model.generate(x, temp=0.8)):
        mx.eval(t)
        token_id = t.item()
        tokens.append(token_id)
        print(tokenizer.decode([token_id]), end="", flush=True)

    print()


if __name__ == "__main__":
    main()
