# Quick Start: Running All Tests

## Single Command to Run All Tests

```bash
cd tmp/rez
python run_all_tests.py
```

This will run all test suites in sequence:
1. **Thermodynamic Grammar** (`test_grammar.py`)
2. **Refactored Architecture** (`test_refactored.py`)
3. **Original Synthesis** (`test_manifold.py`)

## Individual Test Commands

### Grammar Tests
```bash
python test_grammar.py
```
Tests thermodynamic grammar (energy flow through transition matrix).

### Refactored Architecture Tests
```bash
python test_refactored.py
```
Tests the split architecture (SpectralManifold vs SemanticManifold).

### Original Synthesis Tests
```bash
python test_manifold.py
```
Tests:
- Diffusion chord generation
- Next token prediction
- Semantic bridge (concept → audio)
- Differentiable physics training

## Expected Output

When running `python run_all_tests.py`, you should see:

```
================================================================================
MANIFOLD ENGINE - COMPLETE TEST SUITE
================================================================================

[Test 1: Grammar]
[Test 2: Refactored]
[Test 3: Synthesis]

================================================================================
FINAL SUMMARY
================================================================================

Test Results:
  ✓ PASSED: Thermodynamic Grammar (Bond Topology)
  ✓ PASSED: Refactored Architecture (Spectral/Semantic Split)
  ✓ PASSED: Original Synthesis (Diffusion, Next Token, Semantic Bridge, Training)

Total: 3/3 test suites passed

🎉 All tests passed!
```

## What Each Test Demonstrates

### 1. Thermodynamic Grammar
- ✅ Energy flows through transition matrix
- ✅ Sequence-aware predictions (not just resonance)
- ✅ Multi-step thinking ahead

### 2. Refactored Architecture
- ✅ Domain separation (audio vs LLM)
- ✅ Domain-specific distance metrics
- ✅ Clean architecture (no "God Object")

### 3. Original Synthesis
- ✅ Diffusion generation (perfect locking: 0.03 Hz error)
- ✅ Next token prediction (concept → logits)
- ✅ Semantic bridge (concept → audio)
- ✅ Differentiable training (learnable bridge matrix)

## Troubleshooting

If a test fails:
1. Check that all dependencies are installed (`torch`, `tensordict`)
2. Verify files exist: `physics.py`, `semantic.py`, `manifold.py`
3. Check Python version (3.8+)

## Next Steps

After running tests successfully:
1. Explore individual test files to understand each component
2. Modify parameters in test files to experiment
3. Read documentation:
   - `THERMODYNAMIC_GRAMMAR.md` - Grammar implementation
   - `REFACTORING_ROADMAP.md` - Architecture details
   - `DIFFERENTIABLE_PHYSICS.md` - Training details
