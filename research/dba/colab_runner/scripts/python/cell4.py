# Discover Checkpoints
from pathlib import Path

checkpoint_dir = Path(DRIVE_CHECKPOINT_DIR)

if not checkpoint_dir.exists():
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

checkpoints = list(checkpoint_dir.rglob("*.pt"))
print(f"Found {len(checkpoints)} checkpoint(s):")
for ckpt in checkpoints:
    size_mb = ckpt.stat().st_size / 1e6
    print(f"  - {ckpt.name} ({size_mb:.0f} MB)")
