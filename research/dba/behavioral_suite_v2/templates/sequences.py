"""
Sequence templates.

Tests pattern recognition and extrapolation in numeric and
letter sequences.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
import string

from .base import (
    TestCase,
    TestTemplate,
    Difficulty,
    TargetPosition,
    EvalKind,
)


class ArithmeticSequenceTemplate(TestTemplate):
    """Arithmetic sequences: constant difference."""

    category = "sequences"
    subcategory = "arithmetic"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            start = rng.randint(1, 10)
            diff = rng.randint(1, 5)
            length = 4
        elif difficulty == Difficulty.MEDIUM:
            start = rng.randint(1, 20)
            diff = rng.randint(2, 10)
            length = 4
        else:
            start = rng.randint(-10, 30)
            diff = rng.randint(-5, 10)
            if diff == 0:
                diff = 3
            length = 5

        sequence = [start + i * diff for i in range(length)]
        next_val = start + length * diff

        seq_str = ", ".join(map(str, sequence))
        prompt = f"What comes next in the sequence: {seq_str}, ?"
        expected = str(next_val)

        return TestCase(
            id=f"seq_arith_{start}_{diff}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"start": start, "difference": diff, "type": "arithmetic"},
        )


class GeometricSequenceTemplate(TestTemplate):
    """Geometric sequences: constant ratio."""

    category = "sequences"
    subcategory = "geometric"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            start = rng.randint(1, 5)
            ratio = 2
            length = 4
        elif difficulty == Difficulty.MEDIUM:
            start = rng.randint(1, 5)
            ratio = rng.randint(2, 4)
            length = 4
        else:
            start = rng.randint(1, 10)
            ratio = rng.randint(2, 5)
            length = 5

        sequence = [start * (ratio ** i) for i in range(length)]
        next_val = start * (ratio ** length)

        seq_str = ", ".join(map(str, sequence))
        prompt = f"What comes next in the sequence: {seq_str}, ?"
        expected = str(next_val)

        return TestCase(
            id=f"seq_geom_{start}_{ratio}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"start": start, "ratio": ratio, "type": "geometric"},
        )


class LetterSequenceTemplate(TestTemplate):
    """Letter sequences: alphabetic patterns."""

    category = "sequences"
    subcategory = "letters"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        alphabet = string.ascii_uppercase

        if difficulty == Difficulty.EASY:
            # Simple consecutive: A, B, C, D, ?
            start_idx = rng.randint(0, 20)
            length = 4
            step = 1
        elif difficulty == Difficulty.MEDIUM:
            # Skip pattern: A, C, E, G, ?
            start_idx = rng.randint(0, 15)
            length = 4
            step = 2
        else:
            # Larger skip: A, D, G, J, ?
            start_idx = rng.randint(0, 10)
            length = 4
            step = 3

        sequence = [alphabet[(start_idx + i * step) % 26] for i in range(length)]
        next_letter = alphabet[(start_idx + length * step) % 26]

        seq_str = ", ".join(sequence)
        prompt = f"What comes next in the sequence: {seq_str}, ?"
        expected = next_letter

        return TestCase(
            id=f"seq_letter_{sequence[0]}_{step}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"start": sequence[0], "step": step, "type": "letter"},
        )


class FibonacciLikeTemplate(TestTemplate):
    """Fibonacci-like sequences: each term is sum of previous two."""

    category = "sequences"
    subcategory = "fibonacci"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Standard Fibonacci start
            a, b = 1, 1
            length = 5
        elif difficulty == Difficulty.MEDIUM:
            # Different starting values
            a, b = rng.randint(1, 3), rng.randint(1, 3)
            length = 5
        else:
            # Larger starting values
            a, b = rng.randint(2, 5), rng.randint(3, 7)
            length = 6

        sequence = [a, b]
        for _ in range(length - 2):
            sequence.append(sequence[-1] + sequence[-2])
        next_val = sequence[-1] + sequence[-2]

        seq_str = ", ".join(map(str, sequence))
        prompt = f"What comes next in the sequence: {seq_str}, ?"
        expected = str(next_val)

        return TestCase(
            id=f"seq_fib_{a}_{b}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"start": [a, b], "type": "fibonacci"},
        )


class AlternatingSequenceTemplate(TestTemplate):
    """Alternating sequences: two interleaved patterns."""

    category = "sequences"
    subcategory = "alternating"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Simple alternating: 1, 2, 1, 2, ?
            a, b = rng.randint(1, 5), rng.randint(6, 10)
            sequence = [a, b, a, b]
            next_val = a
        elif difficulty == Difficulty.MEDIUM:
            # Alternating with increment: 1, 10, 2, 10, 3, 10, ?
            start = rng.randint(1, 5)
            const = rng.randint(10, 20)
            sequence = []
            for i in range(3):
                sequence.extend([start + i, const])
            next_val = start + 3
        else:
            # Two increasing sequences: 1, 10, 2, 20, 3, 30, ?
            start1 = rng.randint(1, 5)
            start2 = rng.randint(10, 20)
            diff1 = 1
            diff2 = 10
            sequence = []
            for i in range(3):
                sequence.extend([start1 + i * diff1, start2 + i * diff2])
            next_val = start1 + 3 * diff1

        seq_str = ", ".join(map(str, sequence))
        prompt = f"What comes next in the sequence: {seq_str}, ?"
        expected = str(next_val)

        return TestCase(
            id=f"seq_alt_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "alternating"},
        )


class SquareSequenceTemplate(TestTemplate):
    """Square number sequences."""

    category = "sequences"
    subcategory = "squares"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Perfect squares: 1, 4, 9, 16, ?
            start = 1
            length = 4
            sequence = [(start + i) ** 2 for i in range(length)]
            next_val = (start + length) ** 2
        elif difficulty == Difficulty.MEDIUM:
            # Squares with offset: 2, 5, 10, 17, ? (n^2 + 1)
            offset = rng.randint(1, 3)
            length = 4
            sequence = [(i + 1) ** 2 + offset for i in range(length)]
            next_val = (length + 1) ** 2 + offset
        else:
            # Cubes: 1, 8, 27, 64, ?
            start = 1
            length = 4
            sequence = [(start + i) ** 3 for i in range(length)]
            next_val = (start + length) ** 3

        seq_str = ", ".join(map(str, sequence))
        prompt = f"What comes next in the sequence: {seq_str}, ?"
        expected = str(next_val)

        return TestCase(
            id=f"seq_square_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "squares" if difficulty != Difficulty.HARD else "cubes"},
        )


# Export all templates
TEMPLATES = [
    ArithmeticSequenceTemplate(),
    GeometricSequenceTemplate(),
    LetterSequenceTemplate(),
    FibonacciLikeTemplate(),
    AlternatingSequenceTemplate(),
    SquareSequenceTemplate(),
]
