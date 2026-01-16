# Run Benchmark
import os
import sys
from datetime import datetime
from pathlib import Path

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(DRIVE_RESULTS_DIR) / f"behavioral_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

ckpt_args = " ".join([f'"{str(c)}"' for c in checkpoints])

print(f"Output directory: {output_dir}")
print(f"Tests per category: {TESTS_PER_CATEGORY}")
print("\n" + "="*70)
print("Starting benchmark...")
print("="*70 + "\n")

# Change to the correct directory using Colab magic
# This is more reliable than cd in a shell command
%cd /content/caramba/research/dba

# Run as module - Python will find behavioral_suite_v2 in the current directory
!python -m behavioral_suite_v2.multi_checkpoint_eval \
    --checkpoint-files {ckpt_args} \
    --output-dir "{output_dir}" \
    --tests-per-category {TESTS_PER_CATEGORY} \
    --seed {SEED} \
    --max-new-tokens {MAX_NEW_TOKENS} \
    --device cuda \
    --no-browser \
    --verbose

print("\n" + "="*70)
print("Benchmark complete!")
print(f"Results saved to: {output_dir}")
print("="*70)
