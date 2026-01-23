#!/usr/bin/env python3
"""Prepare instruction tuning datasets (Alpaca, OpenAssistant, Dolly).

Downloads and tokenizes instruction datasets with Llama 3.2 tokenizer,
formatting them as User/Assistant pairs with EOS tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


def format_alpaca(example):
    """Format Alpaca example."""
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]
    
    if input_text:
        prompt = f"User: {instruction}\nInput: {input_text}\nAssistant: "
    else:
        prompt = f"User: {instruction}\nAssistant: "
        
    return {"text": prompt + output}


def format_dolly(example):
    """Format Dolly-15k example."""
    instruction = example["instruction"]
    context = example["context"]
    response = example["response"]
    
    if context:
        prompt = f"User: {instruction}\nContext: {context}\nAssistant: "
    else:
        prompt = f"User: {instruction}\nAssistant: "
        
    return {"text": prompt + response}


def format_oasst(example):
    """Format OpenAssistant example (simplified to single turn)."""
    # OASST is a tree, but the top-level dataset is flattened or can be.
    # We'll just look for 'text' field if it's already formatted, or basic structure.
    # Actually, timdettmers/openassistant-guanaco is a good pre-processed version.
    # It has 'text' field with ### Human: ... ### Assistant: ...
    # We will reformat to our User/Assistant convention if needed, or just use it.
    # Let's stick to a consistent format: User: ...\nAssistant: ...
    
    text = example["text"]
    # Simple replace to match our convention (optional, but good for consistency)
    text = text.replace("### Human:", "User:").replace("### Assistant:", "\nAssistant:")
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Prepare instruction datasets")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/datasets/instructions_llama",
        help="Output directory"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Tokenizer model ID"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    
    # Ensure we have an EOS token
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        if "llama-3" in args.model_id.lower():
            eos_token_id = 128001
            print(f"Using fallback EOS ID: {eos_token_id}")
        else:
            raise ValueError("Tokenizer has no EOS token ID")
    print(f"EOS ID: {eos_token_id}")

    all_tokens = []
    
    # 1. Alpaca
    print("\nProcessing Alpaca...")
    ds_alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    for ex in tqdm(ds_alpaca):
        text = format_alpaca(ex)["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(eos_token_id)
        all_tokens.extend(tokens)

    # 2. Dolly-15k
    print("\nProcessing Dolly-15k...")
    ds_dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    for ex in tqdm(ds_dolly):
        text = format_dolly(ex)["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(eos_token_id)
        all_tokens.extend(tokens)

    # 3. OpenAssistant (Guanaco)
    print("\nProcessing OpenAssistant (Guanaco)...")
    ds_oasst = load_dataset("timdettmers/openassistant-guanaco", split="train")
    for ex in tqdm(ds_oasst):
        text = format_oasst(ex)["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Guanaco might already have EOS, but let's be safe and ensure exactly one
        if tokens[-1] != eos_token_id:
            tokens.append(eos_token_id)
        all_tokens.extend(tokens)

    print(f"\nTotal tokens: {len(all_tokens):,}")
    
    # Save as .npy
    out_path = output_dir / "instructions.npy"
    print(f"Saving to {out_path}...")
    np.save(out_path, np.array(all_tokens, dtype=np.uint32))
    
    # Save metadata
    with open(output_dir / "metadata.txt", "w") as f:
        f.write(f"tokenizer: {args.model_id}\n")
        f.write(f"total_tokens: {len(all_tokens)}\n")
        f.write("datasets: alpaca-cleaned, dolly-15k, openassistant-guanaco\n")

    print("Done!")

if __name__ == "__main__":
    main()
