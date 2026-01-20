"""
Unit tests for the resolve module (variable interpolation).
"""
from __future__ import annotations

import unittest
from typing import Any, cast

from config.resolve import Resolver


class TestResolver(unittest.TestCase):
    """Tests for Resolver variable interpolation."""

    def test_resolve_simple_string_var(self) -> None:
        """Resolves a simple string variable."""
        resolver = Resolver({"name": "test"})
        result = resolver.resolve("${name}")
        self.assertEqual(result, "test")

    def test_resolve_int_var(self) -> None:
        """Resolves an integer variable."""
        resolver = Resolver({"count": 42})
        result = resolver.resolve("${count}")
        self.assertEqual(result, 42)

    def test_resolve_float_var(self) -> None:
        """Resolves a float variable."""
        resolver = Resolver({"rate": 0.001})
        result = resolver.resolve("${rate}")
        self.assertEqual(result, 0.001)

    def test_resolve_embedded_var(self) -> None:
        """Resolves embedded variables in a string."""
        resolver = Resolver({"name": "model", "version": "v1"})
        result = resolver.resolve("${name}_${version}.pt")
        self.assertEqual(result, "model_v1.pt")

    def test_resolve_nested_dict(self) -> None:
        """Resolves variables in nested dictionaries."""
        resolver = Resolver({"d_model": 512, "n_heads": 8})
        result = cast(dict[str, Any], resolver.resolve({
            "model": {
                "hidden_size": "${d_model}",
                "attention": {
                    "heads": "${n_heads}",
                }
            }
        }))
        self.assertEqual(result["model"]["hidden_size"], 512)
        self.assertEqual(result["model"]["attention"]["heads"], 8)

    def test_resolve_list(self) -> None:
        """Resolves variables in lists."""
        resolver = Resolver({"a": 1, "b": 2, "c": 3})
        result = resolver.resolve(["${a}", "${b}", "${c}"])
        self.assertEqual(result, [1, 2, 3])

    def test_resolve_tuple(self) -> None:
        """Resolves variables in tuples."""
        resolver = Resolver({"x": 10, "y": 20})
        result = resolver.resolve(("${x}", "${y}"))
        self.assertEqual(result, (10, 20))

    def test_resolve_no_change(self) -> None:
        """Leaves non-variable strings unchanged."""
        resolver = Resolver({})
        result = resolver.resolve("hello world")
        self.assertEqual(result, "hello world")

    def test_resolve_passthrough_types(self) -> None:
        """Non-string primitives are passed through unchanged."""
        resolver = Resolver({})
        self.assertEqual(resolver.resolve(42), 42)
        self.assertEqual(resolver.resolve(3.14), 3.14)
        self.assertEqual(resolver.resolve(True), True)
        self.assertEqual(resolver.resolve(None), None)

    def test_resolve_unknown_var_raises(self) -> None:
        """Unknown variables raise ValueError."""
        resolver = Resolver({"a": 1})
        with self.assertRaises(ValueError) as ctx:
            resolver.resolve("${unknown}")
        self.assertIn("Unknown manifest variable", str(ctx.exception))

    def test_resolve_cycle_raises(self) -> None:
        """Circular variable references raise ValueError."""
        resolver = Resolver({
            "a": "${b}",
            "b": "${a}",
        })
        with self.assertRaises(ValueError) as ctx:
            resolver.resolve("${a}")
        self.assertIn("Cycle detected", str(ctx.exception))

    def test_resolve_transitive_vars(self) -> None:
        """Transitive variable references are resolved."""
        resolver = Resolver({
            "base": 64,
            "doubled": "${base}",  # Note: This will return 64, not 128
        })
        result = resolver.resolve("${doubled}")
        self.assertEqual(result, 64)


if __name__ == "__main__":
    unittest.main()
