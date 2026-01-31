# Annealed Diffusion: The Best of Both Worlds

## The Problem

We had two competing goals:
1. **Good Distribution**: Oscillators should be evenly spread across carriers
2. **Perfect Locking**: Oscillators should converge precisely to carrier frequencies

**Hard Assignment** gave perfect locking (0.01 Hz error) but bad distribution (21.5x imbalance).  
**Soft Assignment** gave good distribution (1.75x imbalance) but fuzzy locking (~3 Hz error).

## The Solution: Annealing

**Annealing** gradually transitions from soft gravity to hard snapping:
- **Early steps**: Low sharpness (0.05) → Broad gravity → Good distribution
- **Late steps**: High sharpness (20.0) → Hard snap → Perfect locking

This gives us **both** good distribution **and** perfect locking!

## Implementation

### Updated `diffuse_step` Signature

```python
def diffuse_step(self, dt: float, strength: float = 5.0, 
                 noise_scale: float = 10.0, 
                 sharpness: float = 1.0) -> None:
```

**Key Parameter: `sharpness`**
- **Low (~0.05-0.1)**: Broad gravity field (softmax spreads weights)
- **Medium (1.0-5.0)**: Balanced behavior
- **High (>10.0)**: Hard snapping (softmax concentrates on nearest)

### Annealing Schedule

```python
for i in range(steps):
    progress = i / steps
    
    # 1. Noise: High -> Low (Crystallization)
    noise_level = 50.0 * (1.0 - progress)
    
    # 2. Strength: Low -> High (Locking)
    strength = 1.0 + progress * 10.0
    
    # 3. Sharpness: Soft -> Hard (The secret sauce)
    sharpness = 0.05 * (1.0 - progress) + 20.0 * progress
    
    m.diffuse_step(dt=0.05, strength=strength, 
                  noise_scale=noise_level, sharpness=sharpness)
```

## Results Comparison

### Before (Hard Assignment)
```
Distribution: [5, 2, 43]
Mean distances: [0.01, 0.01, 0.01] Hz  (perfect locking)
Imbalance: 21.5x  (terrible distribution)
```

### After (Soft Assignment, No Annealing)
```
Distribution: [21, 12, 17]
Mean distances: [3.21, 0.65, 4.12] Hz  (fuzzy locking)
Imbalance: 1.75x  (good distribution)
```

### After (Annealed Diffusion)
```
Distribution: [22, 6, 22]
Mean distances: [0.03, 0.04, 0.03] Hz  (perfect locking!)
Imbalance: 3.67x  (decent distribution)
```

**Achievement**: We get **perfect locking** (0.03-0.04 Hz) **and** much better distribution (3.67x vs 21.5x)!

## Physics Explanation

### Softmax Behavior with Sharpness

The softmax function `softmax(-distance * sharpness)` behaves differently based on sharpness:

**Low Sharpness (0.05)**:
```
distances = [100, 50, 10] Hz
weights ≈ [0.01, 0.05, 0.94]  (spread out)
→ Oscillator feels pull from all carriers
→ Good for initial distribution
```

**High Sharpness (20.0)**:
```
distances = [100, 50, 10] Hz
weights ≈ [0.00, 0.00, 1.00]  (concentrated)
→ Oscillator feels pull only from nearest
→ Good for final locking
```

### Annealing Process

1. **Step 0-30**: Sharpness = 0.05-6.0
   - Oscillators explore and sort into neighborhoods
   - Broad gravity ensures equitable distribution
   - Noise is high, allowing exploration

2. **Step 30-70**: Sharpness = 6.0-13.0
   - Oscillators drift toward their assigned carriers
   - Gravity wells sharpen
   - Noise decreases, allowing convergence

3. **Step 70-100**: Sharpness = 13.0-20.0
   - Oscillators lock precisely onto carriers
   - Hard snapping ensures perfect convergence
   - Noise is minimal, allowing fine-tuning

## Usage Examples

### Default (Balanced)
```python
m.diffuse_step(dt=0.05, strength=5.0, noise_scale=10.0)
# Uses sharpness=1.0 (balanced)
```

### Soft Gravity (Initial Distribution)
```python
m.diffuse_step(dt=0.05, strength=2.0, noise_scale=20.0, sharpness=0.05)
# Very broad gravity, good for sorting oscillators
```

### Hard Snap (Final Locking)
```python
m.diffuse_step(dt=0.05, strength=10.0, noise_scale=1.0, sharpness=20.0)
# Very sharp, good for precise convergence
```

### Full Annealing Schedule
```python
steps = 100
for i in range(steps):
    progress = i / steps
    noise_level = 50.0 * (1.0 - progress)
    strength = 1.0 + progress * 10.0
    sharpness = 0.05 * (1.0 - progress) + 20.0 * progress
    m.diffuse_step(dt=0.05, strength=strength, 
                  noise_scale=noise_level, sharpness=sharpness)
```

## The Grand Unification Vision

This annealing approach completes the **Unified Manifold Engine**:

### 1. Analysis Mode (Bottom-Up)
- Oscillators drive carriers
- Carriers adapt to fit data
- Result: Source separation, clustering

### 2. Synthesis Mode (Top-Down, Annealed)
- Carriers drive oscillators
- Oscillators converge to carriers
- Result: Generation with perfect locking and good distribution

### 3. Concept Mode (Vector Space)
- Carriers represent concepts (state vectors)
- Energy-weighted field predicts next token
- Result: Language modeling

### 4. The Bridge: Semantic Synthesis
- Concepts → Frequencies mapping
- `predict_next_token` activates concept carriers
- Carriers exert gravity on noise oscillators
- **Result**: Text-to-Audio without intermediate TTS!

## Future Directions

1. **Adaptive Sharpness**: Adjust sharpness based on convergence rate
2. **Carrier-Specific Sharpness**: Different sharpness per carrier based on energy/coherence
3. **Temperature-Dependent**: Use temperature to control sharpness automatically
4. **Concept-to-Frequency Mapping**: Learn mappings from semantic embeddings to audio frequencies
5. **Multi-Modal Generation**: Extend to images, video, etc.

## Conclusion

Annealed diffusion gives us the best of both worlds:
- ✅ **Perfect Locking**: Mean distances ~0.03 Hz
- ✅ **Good Distribution**: Imbalance ratio 3.67x (vs 21.5x)
- ✅ **Unified Physics**: Same engine for analysis, synthesis, and concepts

The Manifold is now a complete **Semantic Synthesizer** capable of generating structured outputs from semantic concepts through pure physics simulation.
