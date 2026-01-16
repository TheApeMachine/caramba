# Run Benchmark
import os
import sys
from datetime import datetime
from pathlib import Path

# Set up paths
os.chdir("/content/caramba/research/dba")
sys.path.insert(0, "/content/caramba")
sys.path.insert(0, "/content/caramba/research/dba")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(DRIVE_RESULTS_DIR) / f"behavioral_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

ckpt_args = " ".join([f'"{str(c)}"' for c in checkpoints])

print(f"Output directory: {output_dir}")
print(f"Tests per category: {TESTS_PER_CATEGORY}")
print("\n" + "="*70)
print("Starting benchmark...")
print("="*70 + "\n")

# Run script directly - avoid module resolution issues
!cd /content/caramba/research/dba && PYTHONPATH=/content/caramba:/content/caramba/research/dba python behavioral_suite_v2/multi_checkpoint_eval.py \
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
