"""
Consistency check templates.

Tests that verify the same capability produces consistent results
across different surface-level formats and presentations.
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


class AdditionFormatConsistencyTemplate(TestTemplate):
    """Same addition problem in different formats."""

    category = "consistency_checks"
    subcategory = "addition_format"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        a = rng.randint(1, 50)
        b = rng.randint(1, 50)
        result = a + b

        formats = [
            f"{a} + {b} =",
            f"What is {a} + {b}?",
            f"Calculate {a} plus {b}",
            f"{a} plus {b} equals",
            f"Sum of {a} and {b}:",
            f"Add {a} and {b}:",
        ]

        prompt = rng.choice(formats)
        expected = str(result)

        return TestCase(
            id=f"consist_add_{a}_{b}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"operands": [a, b], "type": "addition_format"},
        )


class CopyFormatConsistencyTemplate(TestTemplate):
    """Same copy task in different formats."""

    category = "consistency_checks"
    subcategory = "copy_format"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            content = rng.choice(["ABC", "123", "XYZ"])
        elif difficulty == Difficulty.MEDIUM:
            content = "".join(rng.choices("ABCDEFGH", k=5))
        else:
            content = "".join(rng.choices("ABCDEFGHIJ0123456789", k=7))

        formats = [
            f"Copy: {content}",
            f"Repeat this: {content}",
            f"Echo: {content}",
            f"{content} ->",
            f"Output: {content}",
            f"Return: {content}",
        ]

        prompt = rng.choice(formats)
        expected = content

        return TestCase(
            id=f"consist_copy_{content[:5]}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"content": content, "type": "copy_format"},
        )


class QuestionAnswerConsistencyTemplate(TestTemplate):
    """Same question in different phrasings."""

    category = "consistency_checks"
    subcategory = "qa_format"

    QUESTIONS = [
        # (variations, answer)
        (
            [
                "What is the capital of France?",
                "Capital of France:",
                "France's capital is",
                "Name the capital of France",
                "Which city is France's capital?",
            ],
            "Paris"
        ),
        (
            [
                "How many days in a week?",
                "Days per week:",
                "A week has how many days?",
                "Number of days in a week:",
                "Count of days in one week:",
            ],
            "7"
        ),
        (
            [
                "What color is the sky?",
                "The sky is what color?",
                "Color of the sky:",
                "Sky color:",
                "What is the sky's color?",
            ],
            "blue"
        ),
    ]

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        variations, answer = rng.choice(self.QUESTIONS)
        prompt = rng.choice(variations)

        return TestCase(
            id=f"consist_qa_{answer}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=answer,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"answer": answer, "type": "qa_format"},
        )


class ListOrderConsistencyTemplate(TestTemplate):
    """Operations that should be order-independent."""

    category = "consistency_checks"
    subcategory = "list_order"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Sum of numbers (order independent)
            numbers = [rng.randint(1, 10) for _ in range(3)]
            rng.shuffle(numbers)
            prompt = f"What is the sum of {', '.join(map(str, numbers))}?"
            expected = str(sum(numbers))

        elif difficulty == Difficulty.MEDIUM:
            # Count items (order independent)
            items = rng.sample(["apple", "banana", "cherry", "date"], 4)
            rng.shuffle(items)
            prompt = f"How many items are in this list: {', '.join(items)}?"
            expected = str(len(items))

        else:
            # Product of numbers
            numbers = [rng.randint(2, 5) for _ in range(3)]
            rng.shuffle(numbers)
            product = 1
            for n in numbers:
                product *= n
            prompt = f"What is the product of {', '.join(map(str, numbers))}?"
            expected = str(product)

        return TestCase(
            id=f"consist_order_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "list_order"},
        )


class SymbolicEquivalenceTemplate(TestTemplate):
    """Equivalent symbolic representations."""

    category = "consistency_checks"
    subcategory = "symbolic_equiv"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Different multiplication symbols
            a, b = rng.randint(2, 10), rng.randint(2, 10)
            result = a * b
            symbols = [
                f"{a} × {b} =",
                f"{a} * {b} =",
                f"{a} x {b} =",
                f"Multiply {a} by {b}:",
            ]
            prompt = rng.choice(symbols)
            expected = str(result)

        elif difficulty == Difficulty.MEDIUM:
            # Different division symbols
            b = rng.randint(2, 10)
            result = rng.randint(2, 10)
            a = b * result
            symbols = [
                f"{a} ÷ {b} =",
                f"{a} / {b} =",
                f"Divide {a} by {b}:",
                f"{a} divided by {b} =",
            ]
            prompt = rng.choice(symbols)
            expected = str(result)

        else:
            # Equivalent expressions
            a = rng.randint(5, 15)
            expressions = [
                (f"{a} + 0 =", str(a)),
                (f"{a} × 1 =", str(a)),
                (f"{a} - 0 =", str(a)),
                (f"{a} ÷ 1 =", str(a)),
            ]
            prompt, expected = rng.choice(expressions)

        return TestCase(
            id=f"consist_symbol_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "symbolic_equiv"},
        )


class ContextStyleConsistencyTemplate(TestTemplate):
    """Same facts presented in different styles."""

    category = "consistency_checks"
    subcategory = "context_style"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        name = rng.choice(["Alice", "Bob", "Charlie", "Diana"])
        age = rng.randint(20, 50)
        city = rng.choice(["Paris", "London", "Tokyo", "Berlin"])

        if difficulty == Difficulty.EASY:
            # Different sentence structures
            styles = [
                f"{name} is {age} years old. How old is {name}?",
                f"{name}, who is {age}, works downtown. What is {name}'s age?",
                f"At {age} years old, {name} is quite accomplished. {name}'s age is?",
            ]
            prompt = rng.choice(styles)
            expected = str(age)

        elif difficulty == Difficulty.MEDIUM:
            # Different fact ordering
            styles = [
                f"{name} lives in {city}. {name} is {age}. Where does {name} live?",
                f"Living in {city}, {name} is {age} years old. {name}'s city:",
                f"{name} ({age}) resides in {city}. Which city does {name} live in?",
            ]
            prompt = rng.choice(styles)
            expected = city

        else:
            # Mixed formats
            job = rng.choice(["engineer", "teacher", "doctor"])
            styles = [
                f"Name: {name}, Age: {age}, Job: {job}. What is {name}'s job?",
                f"{name} works as a {job} and is {age}. {name}'s profession:",
                f"At {age}, {name} is a successful {job}. What does {name} do?",
            ]
            prompt = rng.choice(styles)
            expected = job

        return TestCase(
            id=f"consist_style_{name}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "context_style"},
        )


# Export all templates
TEMPLATES = [
    AdditionFormatConsistencyTemplate(),
    CopyFormatConsistencyTemplate(),
    QuestionAnswerConsistencyTemplate(),
    ListOrderConsistencyTemplate(),
    SymbolicEquivalenceTemplate(),
    ContextStyleConsistencyTemplate(),
]
