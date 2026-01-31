# Manifold Synthesis Updates

## Summary

Fixed the diffusion generation to properly converge by using **nearest-neighbor assignment** instead of Gaussian attraction, which was too weak for far-away oscillators.

## Key Changes

### 1. New `diffuse_step` Method

**Location:** `manifold.py` (around line 1667)

**What it does:**
- Uses **nearest-neighbor assignment** - every oscillator is pulled toward its closest carrier
- Forces bonding by temporarily widening gates to 100.0 radians
- Ensures all oscillators feel a force, even when starting far from carriers

**Key difference from `_apply_restoring_force`:**
- `_apply_restoring_force`: Uses weighted average of bonded carriers (requires existing bonds)
- `diffuse_step`: Uses nearest-neighbor (works even without bonds, forces bonding)

### 2. Updated `predict_next_token` Method

**Location:** `manifold.py` (around line 1610)

**Changes:**
- Now returns `(logits, field_vector)` tuple instead of just `logits`
- Requires carriers to have `state_vector` field (learned embeddings)
- Falls back gracefully if `state_vector` is missing

### 3. Updated `generate_diffusion` Method

**Location:** `manifold.py` (around line 1815)

**Changes:**
- Now uses `diffuse_step` directly instead of calling `step()`
- Implements proper annealing: noise decreases, strength increases over time
- More efficient and converges better

### 4. New Test Suite

**File:** `test_manifold.py`

**Tests:**
1. **Diffusion Chord Generation** - Generates C-Major chord from noise
2. **Next Token Prediction** - Tests language modeling with state vectors

## Usage

### Run the Improved Test Suite

```bash
cd tmp/rez
python test_manifold.py
```

### Expected Output

**Test 1 should show:**
- Total bonds: 50 (all oscillators bonded)
- Mean distances < 50 Hz (oscillators converged to carriers)
- Clear clustering around target frequencies (261, 330, 392 Hz)

**Test 3 should show:**
- Valid logits with reasonable range
- Top 5 predicted tokens
- Carrier energies showing active concepts

### Manual Usage

```python
from manifold import Manifold, ManifoldConfig
import torch

# Create manifold
cfg = ManifoldConfig(dt=0.01, gate_max=10.0)
m = Manifold(cfg)

# Create carriers manually (or use generate_diffusion)
# ... setup carriers and oscillators ...

# Run diffusion steps
for i in range(50):
    noise_level = 20.0 * (1.0 - i / 50)
    strength = 2.0 + (i / 50) * 5.0
    m.diffuse_step(dt=0.05, strength=strength, noise_scale=noise_level)
```

## Why This Works

### The Problem
- Original `_apply_restoring_force` relied on Gaussian attraction matrix
- Gaussian drops to near-zero for far-away oscillators
- Result: oscillators stayed random, mean distance ~1000 Hz

### The Solution
- `diffuse_step` uses **Euclidean distance** to find nearest carrier
- **Every oscillator** feels a force, no matter how far
- **Forced bonding** ensures metrics update correctly
- **Annealing** gradually reduces noise and increases strength

### Physics
1. **Early steps (high noise)**: Oscillators explore, find nearest carrier
2. **Middle steps**: Oscillators drift toward carriers, noise decreases
3. **Late steps (low noise)**: Oscillators lock onto carrier frequencies

## Next Steps

1. **Audio Output**: Convert generated frequencies to audio using STFT/iSTFT
2. **Visualization**: Plot frequency evolution over diffusion steps
3. **Language Model**: Integrate with actual token embeddings and training
4. **Performance**: Optimize for GPU, batch processing

## Files Modified

- `manifold.py` - Added `diffuse_step`, updated `predict_next_token`, updated `generate_diffusion`
- `test_manifold.py` - New comprehensive test suite
- `test_synthesis.py` - Original test suite (still works but less effective)
