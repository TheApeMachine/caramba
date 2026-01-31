#!/usr/bin/env python3
"""
Master Test Runner: Execute all test suites with a single command.

Usage:
    python run_all_tests.py
    
This runs all test suites in sequence and shows a summary at the end.
"""

import sys
import subprocess
import os

def run_test(test_file, description):
    """Run a test file and return success status."""
    print("\n" + "=" * 80)
    print(f"TEST SUITE: {description}")
    print(f"File: {test_file}")
    print("=" * 80 + "\n")
    
    if not os.path.exists(test_file):
        print(f"⚠ Skipping {test_file} (file not found)\n")
        return False
    
    try:
        # Run test and capture output
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=False,  # Show output in real-time
            text=True
        )
        success = result.returncode == 0
        
        print("\n" + "-" * 80)
        if success:
            print(f"✓ {description} - PASSED\n")
        else:
            print(f"✗ {description} - FAILED (exit code: {result.returncode})\n")
        return success
    except Exception as e:
        print(f"\n✗ {description} - ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all test suites."""
    print("\n" + "=" * 80)
    print("MANIFOLD ENGINE - COMPLETE TEST SUITE")
    print("=" * 80)
    print("\nRunning all test suites...")
    
    # Define all test files (in order of complexity)
    tests = [
        ("test_grammar.py", "Thermodynamic Grammar (Bond Topology)"),
        ("test_refactored.py", "Unified Architecture (Spectral + Semantic)"),
    ]
    
    results = []
    
    for test_file, description in tests:
        success = run_test(test_file, description)
        results.append((test_file, description, success))
    
    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, _, s in results if s)
    total = len(results)
    
    print("\nTest Results:")
    for test_file, description, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {description}")
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        print("\nKey Achievements:")
        print("  ✓ Thermodynamic Grammar (energy flow)")
        print("  ✓ Refactored Architecture (domain separation)")
        print("  ✓ Unified Architecture (one system)")
        return 0
    else:
        print(f"\n⚠ {total - passed} test suite(s) failed")
        print("Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())
