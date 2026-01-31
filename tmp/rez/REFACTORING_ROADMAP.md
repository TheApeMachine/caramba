# The Great Decoupling: Refactoring Roadmap

## The Problem: "God Object" Critique

The original `Manifold` class mixed:
- **Audio-specific code** (Hertz, STFT, phase wrapping)
- **LLM-specific code** (embeddings, tokens, logits)
- **Pure physics** (energy, heat, bonds, diffusion)

This created:
- **Namespace clutter**: Methods for audio and LLMs in the same class
- **Safety issues**: Easy to break one domain while fixing another
- **Performance issues**: Can't optimize each domain separately
- **Shallow logic**: No proper grammar/physics for language modeling

## The Solution: Three-Layer Architecture

### Layer 1: `ThermodynamicEngine` (Base Class)
**Pure physics only.** No domain assumptions.

- Knows about: `position`, `energy`, `heat`, `bonds`
- Doesn't know about: Hertz, Tokens, Embeddings, STFT
- Provides: `distance_metric()` hook for domain-specific metrics
- Core methods: `step_physics()`, `_bond_particles()`, `_diffuse_heat()`

### Layer 2: `SpectralManifold` (Audio Domain)
**Audio-specific implementation.**

- Handles: STFT frames, Hertz, phase wrapping
- Distance metric: Log-frequency with duty cycle scaling
- Ingestion: `ingest_frame(freq_bins, magnitudes, phases)`
- No LLM code: No embeddings, no tokens, no grammar

### Layer 3: `SemanticManifold` (LLM Domain)
**Language modeling with thermodynamic grammar.**

- Handles: Token embeddings, positional encodings, logits
- Distance metric: Cosine similarity
- Ingestion: `ingest_tokens(embeddings, positions)`
- **Grammar**: `transition_matrix` for energy flow between concepts
- **Positional encoding**: Makes "Dog" at start ≠ "Dog" at end

## Key Improvements

### 1. Safety
- ✅ `SpectralManifold` doesn't have `transition_matrix`
- ✅ `SemanticManifold` doesn't have `phase`
- ✅ Can't accidentally break one domain while fixing another

### 2. Performance
- ✅ `SpectralManifold` can use 1D optimizations (sorting/binning)
- ✅ `SemanticManifold` can use vector indexes (FAISS/ScaNN)
- ✅ Each domain can optimize its O(N×M) bottleneck differently

### 3. Grammar Fix
- ✅ `SemanticManifold` has `transition_matrix` for concept flow
- ✅ `apply_thermodynamic_grammar()` implements energy flow
- ✅ "The" (high energy) → lowers activation for "Dog"
- ✅ Positional encoding makes sequence position physically distinct

## Architecture Comparison

### Before (God Object)
```python
class Manifold:
    # Audio methods
    def ingest_stft_frame(...)
    def _attraction_matrix(...)  # Uses Hertz
    
    # LLM methods  
    def predict_next_token(...)
    def ingest_embeddings(...)
    
    # Physics (mixed with domain code)
    def step(...)  # Calls audio/LLM-specific code
```

### After (Decoupled)
```python
class ThermodynamicEngine:
    # Pure physics only
    def step_physics(...)
    def distance_metric(...)  # Abstract, overridden
    
class SpectralManifold(ThermodynamicEngine):
    # Audio only
    def ingest_frame(...)
    def distance_metric(...)  # Log-frequency
    
class SemanticManifold(ThermodynamicEngine):
    # LLM only
    def ingest_tokens(...)
    def apply_thermodynamic_grammar(...)
    def distance_metric(...)  # Cosine
```

## Thermodynamic Grammar

### The Problem
Standard language models predict next token by:
- Dot product with vocabulary
- No concept of "grammar" or "flow"

### The Solution
`SemanticManifold` implements grammar as **energy flow**:

```python
# Transition matrix defines grammar rules
transition_matrix[concept_i, concept_j] = probability that j follows i

# Energy flows through the topology
flow = current_energy @ transition_matrix

# Active concepts heat up their grammatical successors
excitation += flow * dt
```

**Example:**
- "The" (high energy) → flows to → "Dog" (lowers activation energy)
- "Dog" (high energy) → flows to → "Barks" (lowers activation energy)
- Result: Grammar emerges from physics, not hard-coded rules

## Positional Encoding

### The Problem
Standard embeddings: "Dog" at position 0 = "Dog" at position 100

### The Solution
`SemanticManifold` adds positional encoding:

```python
position = embedding + positional_encoding(position_index)
```

**Result:**
- "Dog" at start of sentence ≠ "Dog" at end
- Position becomes part of the physical state
- Sequence order affects energy flow and bonding

## Differentiable Physics

### Replaced Hard Gates with Soft Sigmoid

**Before:**
```python
gate_high = (delta_phi.abs() <= gate_width/2).to(float)  # Hard boolean
```

**After:**
```python
gate_scores = torch.sigmoid((gate_width - distances) / (gate_width * 0.1 + eps))  # Soft, differentiable
```

**Benefits:**
- ✅ Fully differentiable (can backprop through physics)
- ✅ Smooth gradients (no discontinuities)
- ✅ Can train bridge matrix end-to-end

## Migration Path

### For Audio Applications
```python
# Old
from manifold import Manifold
m = Manifold(config)

# New (backward compatible)
from manifold_refactored import SpectralManifold
m = SpectralManifold(config, device)
# Or use alias:
from manifold_refactored import Manifold  # Alias to SpectralManifold
```

### For LLM Applications
```python
# New
from manifold_refactored import SemanticManifold
m = SemanticManifold(config, device, embed_dim=128)

# Set up grammar
transition_matrix = learn_or_define_grammar()
m.set_transition_matrix(transition_matrix)

# Ingest tokens
m.ingest_tokens(token_embeddings, positions)
```

## Next Steps

### 1. Complete the Physics Engine
- [ ] Add genesis logic (create attractors from particles)
- [ ] Add mitosis/merge logic
- [ ] Add synthesis mode (diffuse_step)

### 2. Enhance SpectralManifold
- [ ] Implement full STFT pipeline
- [ ] Add phase tracking
- [ ] Optimize with 1D binning

### 3. Enhance SemanticManifold
- [ ] Learn transition matrix from data
- [ ] Add attention-like mechanisms
- [ ] Optimize with vector indexes

### 4. Training Integration
- [ ] End-to-end training for bridge matrix
- [ ] Learn transition matrix from text
- [ ] Joint training of concept → frequency mapping

## Files Created

1. **`manifold_refactored.py`**: New architecture (base + subclasses)
2. **`test_refactored.py`**: Test suite for refactored code
3. **`REFACTORING_ROADMAP.md`**: This document

## Backward Compatibility

The original `manifold.py` remains unchanged. The refactored version:
- ✅ Maintains same physics core
- ✅ Adds domain-specific subclasses
- ✅ Provides alias `Manifold = SpectralManifold` for compatibility
- ✅ Can coexist with original code

## Conclusion

The refactoring addresses all three critiques:

1. ✅ **"God Object"**: Split into base + domain-specific subclasses
2. ✅ **"Shallow Logic"**: Added thermodynamic grammar for LLMs
3. ✅ **"Scalability"**: Each domain can optimize independently

The architecture is now:
- **Safe**: Can't break one domain while fixing another
- **Performant**: Domain-specific optimizations possible
- **Extensible**: Easy to add new domains (e.g., images, 3D)

This sets the foundation for a truly unified physics engine that works across multiple domains while maintaining clean separation of concerns.
