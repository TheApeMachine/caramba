#!/usr/bin/env python3
"""
Prepare instruction tuning datasets for a GPT-2/Tiktoken model.
Fixes the "Yapper" problem by appending explicit <|endoftext|> tokens.
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import tiktoken

# GPT-2 Constants
EOS_TOKEN_ID = 50256  # <|endoftext|>

def format_alpaca(example):
    """Standard Alpaca format."""
    if example["input"]:
        return f"User: {example['instruction']}\n\n{example['input']}\n\nAssistant: {example['output']}"
    return f"User: {example['instruction']}\n\nAssistant: {example['output']}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="artifacts/datasets/instructions_gpt2")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tiktoken encoder: gpt2")
    enc = tiktoken.get_encoding("gpt2")
    
    print(f"Using EOS Token ID: {EOS_TOKEN_ID} (<|endoftext|>)")

    all_tokens = []

    # --- 1. Alpaca ---
    print("\nProcessing Alpaca...")
    try:
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        for i, ex in enumerate(tqdm(ds)):
            text = format_alpaca(ex)
            if i % 100 == 0:
                print(f"\n--- Alpaca Example {i} ---\n{text}\n-------------------")
            # Encode text
            tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
            # 🔴 THE FIX: Append EOS manually
            tokens.append(EOS_TOKEN_ID) 
            all_tokens.extend(tokens)
    except Exception as e:
        print(f"Failed to process Alpaca: {e}")

    # --- 2. OpenAssistant (OASST1) ---
    print("\nProcessing OpenAssistant (OASST1)...")
    try:
        ds_oasst = load_dataset("OpenAssistant/oasst1", split="train")
        # Filter for English to ensure high quality for manners
        ds_oasst = ds_oasst.filter(lambda x: x["lang"] == "en")
        
        # Build message lookup
        msgs = {row["message_id"]: row for row in ds_oasst}
        
        count = 0
        for row in tqdm(ds_oasst):
            # We want (Prompter -> Assistant) pairs.
            # Find assistant messages that reply directly to a prompter.
            if row["role"] == "assistant" and row["parent_id"] in msgs:
                parent = msgs[row["parent_id"]]
                if parent["role"] == "prompter":
                    # Found a pair
                    instruction = parent["text"]
                    response = row["text"]
                    
                    text = f"User: {instruction}\n\nAssistant: {response}"
                    if count % 100 == 0:
                        print(f"\n--- OASST1 Example {count} ---\n{text}\n-------------------")
                    tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
                    tokens.append(EOS_TOKEN_ID)
                    all_tokens.extend(tokens)
                    count += 1

        print(f"  Extracted {count} pairs from OASST1")
        
    except Exception as e:
        print(f"Failed to process OASST1: {e}")

    # Save
    print(f"\nTotal tokens: {len(all_tokens):,}")
    out_path = output_dir / "instructions.npy"
    print(f"Saving to {out_path}...")
    np.save(out_path, np.array(all_tokens, dtype=np.uint32))
    
    # Save metadata
    with open(output_dir / "metadata.txt", "w") as f:
        f.write("tokenizer: tiktoken:gpt2\n")
        f.write(f"vocab_size: 50257\n")
        f.write(f"total_tokens: {len(all_tokens)}\n")
        f.write("datasets: alpaca-cleaned, oasst1\n")
        f.write(f"eos_token_id: {EOS_TOKEN_ID}\n")

    print("Done. Your models will now learn manners.")

if __name__ == "__main__":
    main()
