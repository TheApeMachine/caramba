# Manifold Synthesis Test Commands

## Quick Start

### 1. Run All Synthesis Tests
```bash
cd tmp/rez
python test_synthesis.py
```

This will run:
- **Test 1**: Diffusion chord generation (C-Major from noise)
- **Test 2**: Diffusion melody generation (C-D-E-F-G scale)
- **Test 3**: Next token prediction (language modeling)
- **Test 4**: Audio separation info (requires audio file)

---

## Individual Tests

### Test 1: Generate C-Major Chord from Noise
```python
python -c "
from manifold import Manifold, ManifoldConfig
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = ManifoldConfig(synthesis_mode=True, restoring_force_strength=2.0)
m = Manifold(cfg, device=device)

m.generate_diffusion(
    n_oscillators=50,
    carrier_frequencies=[261.63, 329.63, 392.00],  # C4, E4, G4
    n_steps=100,
    initial_temperature=2.0,
    final_temperature=0.01
)

osc = m.state.get('oscillators')
print(f'Generated {osc.shape[0]} oscillators')
print(f'Frequencies: {osc.get(\"frequency\").cpu().numpy()[:10]}')
"
```

### Test 2: Generate Melody
```python
python -c "
from manifold import Manifold, ManifoldConfig

cfg = ManifoldConfig(synthesis_mode=True)
m = Manifold(cfg)

m.generate_diffusion(
    n_oscillators=100,
    carrier_frequencies=[261.63, 293.66, 329.63, 349.23, 392.00],  # C-D-E-F-G
    n_steps=150,
    initial_temperature=3.0,
    final_temperature=0.005
)
"
```

### Test 3: Next Token Prediction
```python
python -c "
from manifold import Manifold, ManifoldConfig
import torch
from tensordict import TensorDict

cfg = ManifoldConfig()
m = Manifold(cfg)

# Simulate context tokens
signals = TensorDict({
    'frequency': torch.linspace(100, 1000, 10),
    'amplitude': torch.ones(10) * 0.5,
    'phase': torch.rand(10) * 6.28,
    'duration': torch.ones(10)
}, batch_size=[10])

for _ in range(5):
    m.step(signals)

# Predict next token
vocab_embeddings = torch.randn(1000, 128)
vocab_embeddings = vocab_embeddings / (vocab_embeddings.norm(dim=1, keepdim=True) + 1e-8)
logits = m.predict_next_token(vocab_embeddings)
print(f'Top token: {torch.argmax(logits).item()}')
"
```

---

## Audio Separation (Existing Functionality)

### Separate Two Speakers
```bash
cd tmp/rez
python manifold.py two_speakers.wav output_separated/
```

### Separate with Custom Parameters
```python
from manifold import separate_audio_two_speakers_zero_shot

paths = separate_audio_two_speakers_zero_shot(
    wav_path="two_speakers.wav",
    out_dir="output/",
    n_fft=2048,
    hop_length=512,
    target_outputs=2,
)
print(f"Generated: {paths}")
```

---

## Interactive Python Session

```python
# Start interactive session
cd tmp/rez
python

# Then run:
from manifold import Manifold, ManifoldConfig
import torch

# Create manifold in synthesis mode
cfg = ManifoldConfig(
    synthesis_mode=True,
    restoring_force_strength=2.0,
    dt=0.01
)
m = Manifold(cfg)

# Generate a chord
m.generate_diffusion(
    n_oscillators=50,
    carrier_frequencies=[261.63, 329.63, 392.00],  # C-Major
    n_steps=100,
    initial_temperature=2.0,
    final_temperature=0.01
)

# Inspect results
osc = m.state.get('oscillators')
carriers = m.state.get('carriers')
print(f"Oscillators: {osc.shape[0]}")
print(f"Carriers: {carriers.shape[0]}")
print(f"Frequencies: {osc.get('frequency').cpu().numpy()}")
```

---

## Expected Output

### Test 1 Output:
```
============================================================
Test 1: Diffusion Chord Generation
============================================================
Using device: cpu

Generating chord from noise...
Target frequencies: [261.63, 329.63, 392.0] Hz
Carriers: C4, E4, G4

Results:
  Generated oscillators: 50
  Active carriers: 3

Generated frequency distribution:
  Min: 200.00 Hz
  Max: 450.00 Hz
  Mean: 320.00 Hz
  Std: 50.00 Hz

Frequency alignment (distance to nearest carrier):
  Carrier 0 (261.63 Hz): min_dist=5.00 Hz, mean_dist=30.00 Hz
  Carrier 1 (329.63 Hz): min_dist=3.00 Hz, mean_dist=25.00 Hz
  Carrier 2 (392.00 Hz): min_dist=4.00 Hz, mean_dist=28.00 Hz

✓ Test 1 Complete
```

---

## Troubleshooting

### CUDA Out of Memory
- Use CPU: `device = torch.device('cpu')`
- Reduce `n_oscillators` or `n_steps`

### No Carriers Forming
- Check that `carrier_frequencies` is not empty
- Increase `restoring_force_strength`
- Check initial temperature is high enough

### Oscillators Not Converging
- Increase `n_steps`
- Use exponential cooling schedule
- Lower `final_temperature`

---

## Next Steps

1. **Audio Output**: Convert generated frequencies to audio using STFT/iSTFT
2. **Visualization**: Plot frequency evolution over diffusion steps
3. **Language Model**: Integrate with actual token embeddings
4. **Training**: Learn carrier embeddings from data
