"""Tool test generator

Generates deterministic unit test source code for a format decoder tool.
"""

from __future__ import annotations

from dataclasses import dataclass

from lab.unknown_format.oracle import Record


@dataclass(frozen=True, slots=True)
class ToolTestGenerator:
    """Generates `unittest` code that validates decode correctness."""

    def build_tests(self, *, sample_bytes_hex: str, expected_payload_hex: list[str]) -> str:
        """Generate a self-contained `unittest` module as a string."""
        if not isinstance(sample_bytes_hex, str) or not sample_bytes_hex:
            raise ValueError("sample_bytes_hex must be non-empty")
        if not isinstance(expected_payload_hex, list) or not all(isinstance(x, str) for x in expected_payload_hex):
            raise TypeError("expected_payload_hex must be list[str]")
        lines: list[str] = []
        lines.append("import unittest")
        lines.append("")
        lines.append("from tool import decode")
        lines.append("")
        lines.append("class ToolTests(unittest.TestCase):")
        lines.append("    def test_decode(self):")
        lines.append(f"        raw = bytes.fromhex({sample_bytes_hex!r})")
        lines.append("        out = decode(raw)")
        lines.append("        self.assertIsInstance(out, list)")
        lines.append(f"        self.assertEqual(len(out), {len(expected_payload_hex)})")
        for i, hx in enumerate(expected_payload_hex):
            lines.append(f"        self.assertEqual(out[{i}].hex(), {hx!r})")
        lines.append("")
        return "\n".join(lines) + "\n"

