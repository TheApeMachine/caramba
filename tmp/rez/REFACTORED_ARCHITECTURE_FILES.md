# Refactored Architecture: File Guide

## 🎯 The Refactored Architecture (What You Should Focus On)

The refactored architecture consists of **2 core files** + **2 test files**:

### Core Implementation Files

1. **`physics.py`** ⭐ **START HERE**
   - **What it is**: Base physics engine (domain-agnostic)
   - **Key class**: `ThermodynamicEngine`
   - **Purpose**: Pure thermodynamics (energy, heat, bonds, diffusion)
   - **No domain code**: No Hertz, Tokens, Embeddings
   - **Size**: ~120 lines

2. **`semantic.py`** ⭐ **READ THIS SECOND**
   - **What it is**: LLM domain implementation
   - **Key class**: `SemanticManifold` (inherits from `ThermodynamicEngine`)
   - **Purpose**: Language modeling with thermodynamic grammar
   - **Key features**: 
     - Transition matrix (grammar as energy flow)
     - `step_grammar()` - Energy flow through grammar
     - `predict_next()` - Sequence-aware prediction
   - **Size**: ~160 lines

### Test Files

3. **`test_grammar.py`** ⭐ **SEE IT IN ACTION**
   - **What it tests**: Thermodynamic grammar (bond topology)
   - **Demonstrates**: Energy flow through transition matrix
   - **Shows**: Multi-step thinking ahead
   - **Size**: ~200 lines

4. **`test_refactored.py`** ⭐ **ARCHITECTURE VALIDATION**
   - **What it tests**: The split architecture
   - **Demonstrates**: Domain separation (audio vs LLM)
   - **Shows**: Domain-specific distance metrics
   - **Size**: ~220 lines

## 📁 File Relationships

```
physics.py (base class)
    ↑
semantic.py (inherits, adds grammar)
    ↑
test_grammar.py (tests semantic.py)
test_refactored.py (tests both)
```

## 🚫 Files You Can Ignore (For Refactored Architecture)

### Original Architecture (Still Works, But Not Refactored)
- **`manifold.py`** - Original single-class implementation (94KB)
- **`test_manifold.py`** - Tests for original architecture
- **`test_synthesis.py`** - Original synthesis tests

### Legacy/Intermediate Files
- **`manifold_refactored.py`** - Early refactoring attempt (superseded by `physics.py` + `semantic.py`)

## 📖 Reading Order

To understand the refactored architecture:

1. **Read `physics.py`** (10 minutes)
   - Understand the base class
   - See abstract methods

2. **Read `semantic.py`** (15 minutes)
   - See how it extends the base
   - Understand grammar implementation

3. **Run `test_grammar.py`** (5 minutes)
   - See grammar flow in action
   - Understand energy propagation

4. **Run `test_refactored.py`** (5 minutes)
   - See architecture split
   - Understand domain separation

## 🎯 Quick Start

### To Use Refactored Architecture:

```python
# Import the refactored classes
from physics import PhysicsConfig, ThermodynamicEngine
from semantic import SemanticManifold

# Create a semantic manifold (LLM domain)
config = PhysicsConfig(dt=0.01, transition_flux=5.0)
manifold = SemanticManifold(
    config, 
    device=torch.device("cpu"),
    embed_dim=128,
    vocab_size=1000
)

# Use it
manifold.ingest_context(token_embeddings)
manifold.step_grammar()  # Energy flows
logits = manifold.predict_next()  # Sequence-aware prediction
```

### To Run Tests:

```bash
# Test grammar specifically
python test_grammar.py

# Test refactored architecture
python test_refactored.py

# Run all tests
python run_all_tests.py
```

## 📊 File Size Reference

- `physics.py`: 4.1KB (small, focused)
- `semantic.py`: 6.2KB (adds grammar)
- `test_grammar.py`: 6.9KB (comprehensive tests)
- `test_refactored.py`: 8.6KB (architecture tests)

**Total refactored code**: ~26KB (vs 94KB original)

## ✅ Summary

**For the Refactored Architecture, focus on these 4 files:**

1. ✅ `physics.py` - Base class
2. ✅ `semantic.py` - LLM implementation  
3. ✅ `test_grammar.py` - Grammar tests
4. ✅ `test_refactored.py` - Architecture tests

**Everything else is either:**
- Original architecture (still works, but not refactored)
- Documentation
- Legacy files

## 🎓 Key Concepts

### Refactored Architecture Achievements:

1. **Separation of Concerns**
   - `physics.py`: Pure physics (no domain assumptions)
   - `semantic.py`: LLM domain (embeddings, grammar)

2. **Thermodynamic Grammar**
   - Grammar = Energy flow through transition matrix
   - Sequence-aware predictions (not just resonance)
   - Multi-step thinking ahead

3. **Clean Architecture**
   - No "God Object"
   - Domain-specific optimizations possible
   - Easy to extend to new domains
