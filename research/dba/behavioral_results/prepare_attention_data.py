#!/usr/bin/env python3
"""Prepare attention data for the dashboard."""
import json
import os
from pathlib import Path

ATTENTION_DIR = Path("/sessions/determined-clever-sagan/mnt/caramba/research/dba/100k_runs/dba_checkpoint_benchmark/baseline_vs_decoupled/20260115_010942/attention_dump/behavior_sanity")
OUTPUT_FILE = Path("/sessions/determined-clever-sagan/mnt/caramba/research/dba/behavioral_results/attention_samples.json")

def process_attention_case(case_dir):
    """Process a single attention case directory."""
    result = {
        "case_id": case_dir.name,
        "tokens": [],
        "teacher": None,
        "student": None
    }
    
    # Load tokens
    tokens_file = case_dir / "tokens.json"
    if tokens_file.exists():
        with open(tokens_file) as f:
            result["tokens"] = json.load(f)
    
    # Load case info
    case_file = case_dir / "case.json"
    if case_file.exists():
        with open(case_file) as f:
            case_info = json.load(f)
            result.update(case_info)
    
    # Load attention for teacher and student
    for model_type in ["teacher", "student"]:
        attn_file = case_dir / model_type / "attn.json"
        if attn_file.exists():
            with open(attn_file) as f:
                attn_data = json.load(f)
                # Extract first 4 layers, first 4 heads for visualization
                simplified = []
                for layer in attn_data["layers"][:4]:
                    layer_data = {
                        "index": layer["index"],
                        "mode": layer.get("mode", "unknown"),
                        "heads": []
                    }
                    matrices = layer.get("attn", {}).get("matrices", [])
                    for head_idx, matrix in enumerate(matrices[:4]):
                        layer_data["heads"].append({
                            "head_idx": head_idx,
                            "matrix": matrix  # seq_len x seq_len attention weights
                        })
                    simplified.append(layer_data)
                result[model_type] = simplified
    
    return result

def main():
    samples = []
    
    if ATTENTION_DIR.exists():
        for case_dir in sorted(ATTENTION_DIR.iterdir()):
            if case_dir.is_dir():
                try:
                    sample = process_attention_case(case_dir)
                    if sample["tokens"] and (sample["teacher"] or sample["student"]):
                        samples.append(sample)
                except Exception as e:
                    print(f"Error processing {case_dir}: {e}")
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"samples": samples}, f, indent=2)
    
    print(f"Processed {len(samples)} attention samples")

if __name__ == "__main__":
    main()
