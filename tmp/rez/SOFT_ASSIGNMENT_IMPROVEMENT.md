# Soft Assignment Improvement

## Summary

Upgraded `diffuse_step` from **Hard Assignment** (nearest-neighbor) to **Soft Assignment** (weighted gravity field) to prevent mode collapse and achieve more equitable distribution of oscillators across carriers.

## The Problem: Mode Collapse

With hard assignment (nearest-neighbor), oscillators snap instantly to their closest carrier. This causes:

1. **Uneven Distribution**: One carrier captures most oscillators
2. **Geometric Bias**: Carriers at the edges of the frequency range get fewer oscillators
3. **Example**: With targets [261, 329, 392] Hz and noise from 50-2050 Hz:
   - Carrier 2 (392 Hz) captures ~85% of oscillators
   - Carrier 1 (329 Hz) captures ~3%
   - Carrier 0 (261 Hz) captures ~12%

## The Solution: Soft Assignment

Soft assignment uses a **weighted gravity field** where each oscillator feels a pull from **all carriers**, weighted by distance:

```python
# Calculate weights using softmax over distances
weights = torch.softmax(-abs_dists * gravity_sharpness, dim=1)  # [N, M]

# Target is weighted average of all carrier frequencies
target_f = (weights * f_car).sum(dim=1)  # [N]
```

### Key Parameters

- **`gravity_sharpness`**: Controls how "sharp" the gravity wells are
  - **Lower (0.01-0.05)**: Broader gravity, better distribution, slower convergence
  - **Higher (0.1-1.0)**: Sharper gravity, faster convergence, more like hard assignment
  - **Default**: `0.05` (good balance)

- **`soft_assignment`**: Boolean flag to toggle between modes
  - **True (default)**: Use soft assignment (weighted average)
  - **False**: Use hard assignment (nearest-neighbor)

## Results

### Before (Hard Assignment)
```
Distribution: [5, 2, 43]
Imbalance ratio: 21.50x
```

### After (Soft Assignment)
```
Distribution: [21, 12, 17]
Imbalance ratio: 1.75x
```

**Improvement**: 12x better balance!

## Usage

### Default (Soft Assignment)
```python
m.diffuse_step(dt=0.05, strength=5.0, noise_scale=10.0)
# Uses soft_assignment=True, gravity_sharpness=0.05 by default
```

### Custom Gravity Sharpness
```python
# Broader gravity wells (better distribution)
m.diffuse_step(dt=0.05, strength=5.0, noise_scale=10.0, 
              gravity_sharpness=0.01)

# Sharper gravity wells (faster convergence)
m.diffuse_step(dt=0.05, strength=5.0, noise_scale=10.0, 
              gravity_sharpness=0.1)
```

### Hard Assignment (Original Behavior)
```python
m.diffuse_step(dt=0.05, strength=5.0, noise_scale=10.0, 
              soft_assignment=False)
```

## Physics Explanation

### Hard Assignment (Nearest-Neighbor)
- Each oscillator has a **single target**: its closest carrier
- Force: `F = -k * (x - target_nearest)`
- **Problem**: Oscillators in the middle feel no pull from distant carriers

### Soft Assignment (Gravity Field)
- Each oscillator has a **weighted target**: average of all carriers
- Weights: `w_i = softmax(-distance_i * sharpness)`
- Force: `F = -k * (x - Σ(w_i * carrier_i))`
- **Benefit**: Oscillators feel pull from all carriers, preventing mode collapse

## When to Use Each Mode

### Use Soft Assignment (Default) When:
- ✅ You want balanced distribution across carriers
- ✅ Carriers are evenly spaced in frequency
- ✅ You're generating music/chords (want all notes present)
- ✅ You want smoother convergence

### Use Hard Assignment When:
- ✅ You want fastest convergence
- ✅ You're doing analysis (not generation)
- ✅ You have many carriers and want clear separation
- ✅ Performance is critical

## Implementation Details

The soft assignment is implemented in `manifold.py` in the `diffuse_step` method:

```python
def diffuse_step(self, dt: float, strength: float = 5.0, 
                 noise_scale: float = 10.0, 
                 soft_assignment: bool = True, 
                 gravity_sharpness: float = 0.05) -> None:
    # ... distance calculation ...
    
    if soft_assignment:
        weights = torch.softmax(-abs_dists * gravity_sharpness, dim=1)
        target_f = (weights * f_car).sum(dim=1)
    else:
        _, nearest_idx = abs_dists.min(dim=1)
        target_f = carriers.get("frequency")[nearest_idx]
    
    # ... apply force and update ...
```

## Test Results

Running `test_manifold.py` with soft assignment:

```
Frequency alignment:
  Carrier 0 (261.63 Hz): count=21 (42%), mean_dist=3.21 Hz
  Carrier 1 (329.63 Hz): count=12 (24%), mean_dist=0.65 Hz
  Carrier 2 (392.00 Hz): count=17 (34%), mean_dist=4.12 Hz

Distribution analysis:
  Expected uniform: ~16 oscillators per carrier
  Actual distribution: [21, 12, 17]
  Imbalance ratio: 1.75x (much better than 21.50x!)
```

## Future Improvements

1. **Adaptive Sharpness**: Start with low sharpness (broad), increase over time (sharpen)
2. **Carrier-Specific Sharpness**: Different sharpness per carrier based on energy/coherence
3. **Temperature-Dependent**: Use temperature to control sharpness (hot = broad, cold = sharp)
4. **Energy-Weighted**: Weight carriers by their energy in addition to distance
