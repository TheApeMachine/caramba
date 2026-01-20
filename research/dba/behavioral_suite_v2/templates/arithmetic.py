"""
Arithmetic templates.

Tests numerical reasoning with addition, subtraction, multiplication,
division, and more complex operations.

NOTE: These templates use CHOICE_LOGPROB scoring for clean capability measurement.
For integer results, choices include the correct answer plus nearby wrong answers
to test precise numerical reasoning without decoder confounds.
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


def _generate_int_choices(correct: int, rng: random.Random, num_distractors: int = 3) -> list[str]:
    """Generate integer choice set with correct answer and plausible distractors."""
    distractors = set()

    # Add off-by-one errors
    distractors.add(correct + 1)
    distractors.add(correct - 1)

    # Add off-by-ten errors (common arithmetic mistakes)
    distractors.add(correct + 10)
    distractors.add(correct - 10)

    # Add some random nearby values
    for _ in range(10):
        offset = rng.randint(-20, 20)
        if offset != 0:
            distractors.add(correct + offset)

    # Remove the correct answer if it accidentally got added
    distractors.discard(correct)

    # Sample the required number of distractors
    distractors = list(distractors)
    rng.shuffle(distractors)
    selected_distractors = distractors[:num_distractors]

    # Build choices with leading space for token alignment
    choices = [f" {correct}"] + [f" {d}" for d in selected_distractors]
    rng.shuffle(choices)

    return choices


class AdditionTemplate(TestTemplate):
    """Addition problems with varying difficulty."""

    category = "arithmetic"
    subcategory = "addition"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            a = rng.randint(1, 9)
            b = rng.randint(1, 9)
        elif difficulty == Difficulty.MEDIUM:
            a = rng.randint(10, 99)
            b = rng.randint(10, 99)
        else:
            a = rng.randint(100, 999)
            b = rng.randint(100, 999)

        result = a + b

        # Pure few-shot equation format for base models
        ex1 = (rng.randint(1, 9), rng.randint(1, 9))
        ex2 = (rng.randint(1, 9), rng.randint(1, 9))
        prompt = f"{ex1[0]} + {ex1[1]} = {ex1[0] + ex1[1]}\n{ex2[0]} + {ex2[1]} = {ex2[0] + ex2[1]}\n{a} + {b} ="

        choices = _generate_int_choices(result, rng)
        expected = f" {result}"

        return TestCase(
            id=f"math_add_{a}_{b}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"operands": [a, b], "operation": "add"},
        )


class SubtractionTemplate(TestTemplate):
    """Subtraction problems."""

    category = "arithmetic"
    subcategory = "subtraction"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            a = rng.randint(5, 15)
            b = rng.randint(1, a)  # Ensure positive result
        elif difficulty == Difficulty.MEDIUM:
            a = rng.randint(20, 99)
            b = rng.randint(10, a)
        else:
            # Allow negative results
            a = rng.randint(10, 100)
            b = rng.randint(10, 150)

        result = a - b

        # Pure few-shot equation format for base models
        ex1 = (rng.randint(5, 15), rng.randint(1, 5))
        ex2 = (rng.randint(10, 20), rng.randint(1, 10))
        prompt = f"{ex1[0]} - {ex1[1]} = {ex1[0] - ex1[1]}\n{ex2[0]} - {ex2[1]} = {ex2[0] - ex2[1]}\n{a} - {b} ="

        choices = _generate_int_choices(result, rng)
        expected = f" {result}"

        return TestCase(
            id=f"math_sub_{a}_{b}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"operands": [a, b], "operation": "sub"},
        )


class MultiplicationTemplate(TestTemplate):
    """Multiplication problems."""

    category = "arithmetic"
    subcategory = "multiplication"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            a = rng.randint(2, 9)
            b = rng.randint(2, 9)
        elif difficulty == Difficulty.MEDIUM:
            a = rng.randint(10, 20)
            b = rng.randint(2, 9)
        else:
            a = rng.randint(10, 30)
            b = rng.randint(10, 30)

        result = a * b

        # Pure few-shot equation format for base models
        ex1 = (rng.randint(2, 9), rng.randint(2, 9))
        ex2 = (rng.randint(2, 9), rng.randint(2, 9))
        prompt = f"{ex1[0]} * {ex1[1]} = {ex1[0] * ex1[1]}\n{ex2[0]} * {ex2[1]} = {ex2[0] * ex2[1]}\n{a} * {b} ="

        choices = _generate_int_choices(result, rng)
        expected = f" {result}"

        return TestCase(
            id=f"math_mul_{a}_{b}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"operands": [a, b], "operation": "mul"},
        )


class DivisionTemplate(TestTemplate):
    """Division problems (exact division)."""

    category = "arithmetic"
    subcategory = "division"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            b = rng.randint(2, 9)
            result = rng.randint(1, 9)
            a = b * result  # Ensure exact division
        elif difficulty == Difficulty.MEDIUM:
            b = rng.randint(2, 12)
            result = rng.randint(5, 15)
            a = b * result
        else:
            b = rng.randint(5, 20)
            result = rng.randint(10, 25)
            a = b * result

        # Pure few-shot equation format for base models
        ex1_b = rng.randint(2, 9)
        ex1_r = rng.randint(1, 9)
        ex1_a = ex1_b * ex1_r
        ex2_b = rng.randint(2, 9)
        ex2_r = rng.randint(1, 9)
        ex2_a = ex2_b * ex2_r
        prompt = f"{ex1_a} / {ex1_b} = {ex1_r}\n{ex2_a} / {ex2_b} = {ex2_r}\n{a} / {b} ="

        choices = _generate_int_choices(result, rng)
        expected = f" {result}"

        return TestCase(
            id=f"math_div_{a}_{b}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"operands": [a, b], "operation": "div"},
        )


class ChainedOperationsTemplate(TestTemplate):
    """Multiple operations in sequence."""

    category = "arithmetic"
    subcategory = "chained"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # a + b + c
            a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
            result = a + b + c
            prompt = f"{a} + {b} + {c} ="
        elif difficulty == Difficulty.MEDIUM:
            # a + b - c or a * b + c
            a, b = rng.randint(5, 15), rng.randint(1, 10)
            c = rng.randint(1, 5)
            if rng.random() < 0.5:
                result = a + b - c
                prompt = f"{a} + {b} - {c} ="
            else:
                result = a * b + c
                prompt = f"{a} * {b} + {c} ="
        else:
            # More complex: (a + b) * c
            a, b = rng.randint(2, 10), rng.randint(2, 10)
            c = rng.randint(2, 5)
            result = (a + b) * c
            prompt = f"({a} + {b}) * {c} ="

        choices = _generate_int_choices(result, rng)
        expected = f" {result}"

        return TestCase(
            id=f"math_chain_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"operation": "chained"},
        )


class WordProblemTemplate(TestTemplate):
    """Arithmetic word problems."""

    category = "arithmetic"
    subcategory = "word_problem"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        names = ["Alice", "Bob", "Charlie", "Diana"]
        name = rng.choice(names)

        if difficulty == Difficulty.EASY:
            a = rng.randint(2, 10)
            b = rng.randint(1, 5)
            templates = [
                (f"{name} has {a} apples. She gets {b} more. How many apples does she have?", a + b),
                (f"{name} has {a} cookies and eats {b}. How many are left?", a - b),
            ]
        elif difficulty == Difficulty.MEDIUM:
            a = rng.randint(5, 15)
            b = rng.randint(2, 5)
            templates = [
                (f"{name} has {a} dollars. She earns {b} dollars per hour for 3 hours. How much does she have now?", a + b * 3),
                (f"There are {a} students in each of {b} classrooms. How many students total?", a * b),
            ]
        else:
            a = rng.randint(10, 30)
            b = rng.randint(5, 15)
            c = rng.randint(2, 8)
            templates = [
                (f"{name} has {a} dollars. She spends {b} dollars and then earns {c} dollars. How much does she have?", a - b + c),
                (f"A store has {a} items. They sell {b} and receive {c} new ones. How many items now?", a - b + c),
            ]

        prompt, result = rng.choice(templates)
        choices = _generate_int_choices(result, rng)
        expected = f" {result}"

        return TestCase(
            id=f"math_word_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"type": "word_problem"},
        )


class ComparisonArithmeticTemplate(TestTemplate):
    """Compare arithmetic results."""

    category = "arithmetic"
    subcategory = "comparison"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            c, d = rng.randint(1, 9), rng.randint(1, 9)
            left = a + b
            right = c + d
            prompt = f"Which is larger: {a} + {b} or {c} + {d}?"
        elif difficulty == Difficulty.MEDIUM:
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            c, d = rng.randint(2, 9), rng.randint(2, 9)
            left = a * b
            right = c * d
            prompt = f"Which is larger: {a} * {b} or {c} * {d}?"
        else:
            a, b, c = rng.randint(2, 10), rng.randint(2, 10), rng.randint(1, 5)
            d, e = rng.randint(5, 15), rng.randint(2, 8)
            left = a + b * c
            right = d * e
            prompt = f"Which is larger: {a} + {b} * {c} or {d} * {e}?"

        # Use clear first/second/equal choices
        if left > right:
            expected = " first"
        elif right > left:
            expected = " second"
        else:
            expected = " equal"

        choices = [" first", " second", " equal"]

        return TestCase(
            id=f"math_compare_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"type": "comparison", "left_value": left, "right_value": right},
        )


# Export all templates
TEMPLATES = [
    AdditionTemplate(),
    SubtractionTemplate(),
    MultiplicationTemplate(),
    DivisionTemplate(),
    ChainedOperationsTemplate(),
    WordProblemTemplate(),
    ComparisonArithmeticTemplate(),
]
