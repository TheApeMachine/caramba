"""
Edge case templates.

Tests handling of boundary conditions, empty inputs, zeros,
repeated values, and other edge cases.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .base import (
    TestCase,
    TestTemplate,
    Difficulty,
    TargetPosition,
    EvalKind,
)


class ZeroHandlingTemplate(TestTemplate):
    """Operations involving zero."""

    category = "edge_cases"
    subcategory = "zero"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Add/subtract zero
            n = rng.randint(1, 100)
            ops = [
                (f"{n} + 0 =", str(n)),
                (f"0 + {n} =", str(n)),
                (f"{n} - 0 =", str(n)),
            ]
            prompt, expected = rng.choice(ops)

        elif difficulty == Difficulty.MEDIUM:
            # Multiply by zero
            n = rng.randint(1, 100)
            ops = [
                (f"{n} × 0 =", "0"),
                (f"0 × {n} =", "0"),
                (f"0 × 0 =", "0"),
            ]
            prompt, expected = rng.choice(ops)

        else:
            # Division with zero (dividend)
            n = rng.randint(1, 100)
            prompt = f"0 ÷ {n} ="
            expected = "0"

        return TestCase(
            id=f"edge_zero_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "zero"},
        )


class SingleElementTemplate(TestTemplate):
    """Operations on single elements."""

    category = "edge_cases"
    subcategory = "single_element"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Single character copy
            char = rng.choice("ABCXYZ")
            prompt = f"Copy this: {char}"
            expected = char

        elif difficulty == Difficulty.MEDIUM:
            # Single digit operations
            n = rng.randint(0, 9)
            ops = [
                (f"What is the first digit of {n}?", str(n)),
                (f"What is the last digit of {n}?", str(n)),
                (f"How many digits in {n}?", "1"),
            ]
            prompt, expected = rng.choice(ops)

        else:
            # Single item list
            item = rng.choice(["apple", "banana", "cherry"])
            ops = [
                (f"What is the first item in this list: {item}?", item),
                (f"What is the last item in this list: {item}?", item),
                (f"How many items in this list: {item}?", "1"),
            ]
            prompt, expected = rng.choice(ops)

        return TestCase(
            id=f"edge_single_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "single_element"},
        )


class RepeatedValuesTemplate(TestTemplate):
    """Handling of repeated values."""

    category = "edge_cases"
    subcategory = "repeated"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # All same character
            char = rng.choice("XYZ")
            count = rng.randint(3, 5)
            seq = char * count
            prompt = f"Copy this: {seq}"
            expected = seq

        elif difficulty == Difficulty.MEDIUM:
            # Repeated number in sequence
            n = rng.randint(1, 9)
            count = rng.randint(4, 6)
            seq = ", ".join([str(n)] * count)
            prompt = f"What comes next in the sequence: {seq}, ?"
            expected = str(n)

        else:
            # Count repeated items
            item = rng.choice(["a", "x", "1"])
            count = rng.randint(5, 10)
            seq = " ".join([item] * count)
            prompt = f"How many items are in this list: {seq}?"
            expected = str(count)

        return TestCase(
            id=f"edge_repeat_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "repeated"},
        )


class LargeNumbersTemplate(TestTemplate):
    """Operations with large numbers."""

    category = "edge_cases"
    subcategory = "large_numbers"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Comparison of large numbers
            a = rng.randint(1000, 9999)
            b = rng.randint(1000, 9999)
            prompt = f"Which is larger: {a} or {b}?"
            expected = str(max(a, b))

        elif difficulty == Difficulty.MEDIUM:
            # Simple operation with large numbers
            a = rng.randint(100, 999)
            b = rng.randint(100, 999)
            result = a + b
            prompt = f"{a} + {b} ="
            expected = str(result)

        else:
            # Multiplication resulting in large number
            a = rng.randint(10, 99)
            b = rng.randint(10, 99)
            result = a * b
            prompt = f"{a} × {b} ="
            expected = str(result)

        return TestCase(
            id=f"edge_large_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "large_numbers"},
        )


class NegativeNumbersTemplate(TestTemplate):
    """Operations with negative numbers."""

    category = "edge_cases"
    subcategory = "negative"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Simple negative
            n = rng.randint(1, 20)
            prompt = f"What is 0 - {n}?"
            expected = str(-n)

        elif difficulty == Difficulty.MEDIUM:
            # Subtraction resulting in negative
            a = rng.randint(1, 10)
            b = rng.randint(a + 1, a + 20)
            prompt = f"{a} - {b} ="
            expected = str(a - b)

        else:
            # Operations with negative numbers
            a = rng.randint(-20, -1)
            b = rng.randint(1, 20)
            ops = [
                (f"{a} + {b} =", str(a + b)),
                (f"{a} × 2 =", str(a * 2)),
                (f"What is negative {abs(a)}?", str(a)),
            ]
            prompt, expected = rng.choice(ops)

        return TestCase(
            id=f"edge_negative_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "negative"},
        )


class BoundaryConditionsTemplate(TestTemplate):
    """Boundary conditions in sequences and ranges."""

    category = "edge_cases"
    subcategory = "boundary"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # First/last letter
            ops = [
                ("What letter comes after Z?", "A"),  # Wrap around
                ("What letter comes before A?", "Z"),  # Wrap around
                ("What is the first letter of the alphabet?", "A"),
                ("What is the last letter of the alphabet?", "Z"),
            ]
            prompt, expected = rng.choice(ops)

        elif difficulty == Difficulty.MEDIUM:
            # Month boundaries
            ops = [
                ("What month comes after December?", "January"),
                ("What month comes before January?", "December"),
                ("What is the 12th month?", "December"),
                ("What is the 1st month?", "January"),
            ]
            prompt, expected = rng.choice(ops)

        else:
            # Day boundaries
            ops = [
                ("What day comes after Saturday?", "Sunday"),
                ("What day comes before Sunday?", "Saturday"),
                ("If today is December 31, what month is tomorrow?", "January"),
                ("What is day 7 of the week?", "Sunday"),
            ]
            prompt, expected = rng.choice(ops)

        return TestCase(
            id=f"edge_boundary_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "boundary"},
        )


class IdentityOperationsTemplate(TestTemplate):
    """Identity operations that should return input unchanged."""

    category = "edge_cases"
    subcategory = "identity"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            n = rng.randint(1, 100)
            ops = [
                (f"{n} + 0 =", str(n)),
                (f"{n} × 1 =", str(n)),
                (f"{n} ÷ 1 =", str(n)),
            ]
            prompt, expected = rng.choice(ops)

        elif difficulty == Difficulty.MEDIUM:
            n = rng.randint(1, 100)
            ops = [
                (f"{n} - 0 =", str(n)),
                (f"1 × {n} =", str(n)),
                (f"{n} ^ 1 =", str(n)),
            ]
            prompt, expected = rng.choice(ops)

        else:
            word = rng.choice(["hello", "world", "test"])
            ops = [
                (f"Reverse the reversal of '{word}':", word),
                (f"What is '{word}' in lowercase (already lowercase)?", word),
            ]
            prompt, expected = rng.choice(ops)

        return TestCase(
            id=f"edge_identity_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "identity"},
        )


# Export all templates
TEMPLATES = [
    ZeroHandlingTemplate(),
    SingleElementTemplate(),
    RepeatedValuesTemplate(),
    LargeNumbersTemplate(),
    NegativeNumbersTemplate(),
    BoundaryConditionsTemplate(),
    IdentityOperationsTemplate(),
]
