#!/usr/bin/env python3
"""Prepare FineWeb-Edu dataset with Llama tokenizer.

Downloads FineWeb-Edu from HuggingFace and tokenizes with Llama 3.2 tokenizer,
saving shards as .npy files for training.

Usage:
    python scripts/prepare_fineweb_llama.py --tokens 1B --output artifacts/datasets/fineweb_llama
    python scripts/prepare_fineweb_llama.py --tokens 10B --output artifacts/datasets/fineweb_llama_10b
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_size(size_str: str) -> int:
    """Parse size string like '1B', '10B', '100M' to number of tokens."""
    size_str = size_str.upper().strip()
    multipliers = {"B": 1_000_000_000, "M": 1_000_000, "K": 1_000}
    for suffix, mult in multipliers.items():
        if size_str.endswith(suffix):
            return int(float(size_str[:-1]) * mult)
    return int(size_str)


def main():
    parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu with Llama tokenizer")
    parser.add_argument(
        "--tokens",
        type=str,
        default="1B",
        help="Number of tokens to generate (e.g., 1B, 10B, 100M)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/datasets/fineweb_llama",
        help="Output directory for tokenized data"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000_000,  # 100M tokens per shard
        help="Tokens per shard file"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID for tokenizer"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset to tokenize"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="sample-10BT",
        help="Dataset subset (sample-10BT, sample-100BT, sample-350BT)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download/tokenization if shards exist, just recombine"
    )
    args = parser.parse_args()

    target_tokens = parse_size(args.tokens)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target tokens: {target_tokens:,}")
    print(f"Output directory: {output_dir}")
    print(f"Model ID: {args.model_id}")
    print(f"Dataset: {args.dataset} ({args.subset})")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size:,}")

    # Determine EOS token
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        # Fallback for Llama 3 if not explicitly set (usually 128001)
        if "llama-3" in args.model_id.lower():
            eos_token_id = 128001
            print(f"Warning: tokenizer.eos_token_id is None, falling back to Llama-3 default: {eos_token_id}")
        else:
            print("Warning: tokenizer.eos_token_id is None, and no fallback available. Documents will not be separated.")
    else:
        print(f"EOS token ID: {eos_token_id}")

    # Load dataset (streaming to avoid memory issues)
    print(f"\nLoading dataset {args.dataset}...")
    dataset = load_dataset(
        args.dataset,
        args.subset,
        split="train",
        streaming=True,
    )

    # Tokenize and save shards
    all_tokens = []
    total_tokens = 0
    shard_idx = 0

    if args.skip_download and any(output_dir.glob("shard_*.npy")):
        print("\nSkipping download/tokenization as requested. Recombining existing shards...")
        # Count shards
        shard_files = sorted(list(output_dir.glob("shard_*.npy")))
        shard_idx = len(shard_files)
        total_tokens = 0 # We'll count during recombination
    else:
        print(f"\nTokenizing... (target: {target_tokens:,} tokens)")

        pbar = tqdm(total=target_tokens, unit="tok", unit_scale=True)

        for example in dataset:
            text = example.get("text", "")
            if not text:
                continue

            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Append EOS token to separate documents
            if eos_token_id is not None:
                tokens.append(eos_token_id)
                
            all_tokens.extend(tokens)
            total_tokens += len(tokens)
            pbar.update(len(tokens))

            # Save shard when full
            while len(all_tokens) >= args.shard_size:
                shard_tokens = all_tokens[:args.shard_size]
                all_tokens = all_tokens[args.shard_size:]

                shard_path = output_dir / f"shard_{shard_idx:05d}.npy"
                np.save(shard_path, np.array(shard_tokens, dtype=np.uint32))
                shard_idx += 1

            # Stop when we have enough
            if total_tokens >= target_tokens:
                break

        pbar.close()

        # Save remaining tokens as final shard
        if all_tokens:
            shard_path = output_dir / f"shard_{shard_idx:05d}.npy"
            np.save(shard_path, np.array(all_tokens, dtype=np.uint32))
            shard_idx += 1

    # Concatenate all shards into single file (optional, for simple loading)
    print(f"\nConcatenating {shard_idx} shards into single file...")
    all_data = []
    
    # Check if we can reuse existing shards instead of re-downloading
    # If shards exist but combined file is missing, we can just recombine.
    # But since we changed the tokenization logic (append_eos), we MUST re-tokenize.
    # So we assume the user has deleted the output directory or we overwrite.
    
    for i in range(shard_idx):
        shard_path = output_dir / f"shard_{i:05d}.npy"
        if not shard_path.exists():
             print(f"Warning: Shard {shard_path} missing during recombination.")
             continue
        all_data.append(np.load(shard_path))

    if not all_data:
        print("Error: No data found to concatenate.")
        return

    combined = np.concatenate(all_data)
    combined_path = output_dir / f"fineweb_llama_{args.tokens.lower()}.npy"
    np.save(combined_path, combined)

    print(f"\nDone!")
    print(f"  Total tokens: {len(combined):,}")
    print(f"  Shards: {shard_idx}")
    print(f"  Combined file: {combined_path}")
    print(f"  Vocab size: {vocab_size}")

    # Save metadata
    meta_path = output_dir / "metadata.txt"
    with open(meta_path, "w") as f:
        f.write(f"tokenizer: {args.model_id}\n")
        f.write(f"vocab_size: {vocab_size}\n")
        f.write(f"total_tokens: {len(combined)}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"subset: {args.subset}\n")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
