# File Structure: Refactored Architecture Guide

## Overview

There are **two architectures** in this codebase:

1. **Original Architecture** (`manifold.py`) - Single "God Object" class
2. **Refactored Architecture** (`physics.py` + `semantic.py`) - Split into base + domain-specific

## Refactored Architecture Files

### Core Files (The Refactored Architecture)

These are the **new, refactored files** that implement the split architecture:

1. **`physics.py`** ⭐ **BASE CLASS**
   - Pure physics engine (domain-agnostic)
   - `ThermodynamicEngine` class
   - No domain-specific code (no Hertz, Tokens, Embeddings)
   - Abstract methods: `compute_distances()`, `compute_targets()`

2. **`semantic.py`** ⭐ **LLM DOMAIN**
   - `SemanticManifold` class (subclass of `ThermodynamicEngine`)
   - Implements thermodynamic grammar
   - Handles embeddings, tokens, transition matrix
   - Methods: `step_grammar()`, `predict_next()`, `learn_transition()`

### Test Files

3. **`test_grammar.py`** ⭐ **GRAMMAR TESTS**
   - Tests thermodynamic grammar (energy flow)
   - Demonstrates bond topology
   - Shows multi-step thinking

4. **`test_refactored.py`** ⭐ **ARCHITECTURE TESTS**
   - Tests the split architecture
   - Tests `SpectralManifold` vs `SemanticManifold`
   - Tests domain-specific distance metrics

### Utility Files

5. **`run_all_tests.py`**
   - Master test runner
   - Runs all test suites

## Original Architecture Files (Still Exists)

These files are **unchanged** and represent the original architecture:

- **`manifold.py`** - Original single-class implementation
- **`test_manifold.py`** - Tests for original architecture
- **`test_synthesis.py`** - Original synthesis tests

## Legacy/Intermediate Files

These were created during development but are **superseded**:

- **`manifold_refactored.py`** - Early refactoring attempt (superseded by `physics.py` + `semantic.py`)

## Quick Reference

### To Use the Refactored Architecture:

```python
# For LLM domain (with grammar)
from semantic import SemanticManifold
from physics import PhysicsConfig

config = PhysicsConfig()
manifold = SemanticManifold(config, device, embed_dim=128, vocab_size=1000)
```

### To Use the Original Architecture:

```python
# Original (still works)
from manifold import Manifold, ManifoldConfig

config = ManifoldConfig()
manifold = Manifold(config, device)
```

## File Dependencies

### Refactored Architecture:
```
physics.py (base)
    ↑
semantic.py (inherits from physics.py)
    ↑
test_grammar.py (tests semantic.py)
test_refactored.py (tests both)
```

### Original Architecture:
```
manifold.py (standalone)
    ↑
test_manifold.py (tests manifold.py)
test_synthesis.py (tests manifold.py)
```

## Which Files Should You Focus On?

### For Refactored Architecture (New):
1. **`physics.py`** - Read this first (base class)
2. **`semantic.py`** - Read this second (LLM implementation)
3. **`test_grammar.py`** - See it in action
4. **`test_refactored.py`** - See architecture split

### For Original Architecture (Legacy):
1. **`manifold.py`** - Original implementation
2. **`test_manifold.py`** - Original tests

## Summary

**Refactored Architecture = 2 core files:**
- `physics.py` (base)
- `semantic.py` (LLM domain)

**Plus 2 test files:**
- `test_grammar.py` (grammar tests)
- `test_refactored.py` (architecture tests)

**Everything else is either:**
- Original architecture (still works)
- Documentation
- Legacy/intermediate files
