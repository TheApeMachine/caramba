"""
Robustness templates.

Tests consistency under variations in phrasing, casing,
whitespace, and other surface-level changes.
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


class RephraseConsistencyTemplate(TestTemplate):
    """Same question, different phrasing - should give same answer."""

    category = "robustness"
    subcategory = "rephrase"

    QUESTION_SETS = [
        # Capital questions
        (
            [
                "What is the capital of France?",
                "What city is the capital of France?",
                "France's capital is which city?",
                "Name the capital city of France.",
            ],
            "Paris"
        ),
        (
            [
                "What is 5 plus 3?",
                "Calculate 5 + 3",
                "5 added to 3 equals what?",
                "What do you get when you add 5 and 3?",
            ],
            "8"
        ),
        (
            [
                "How many days are in a week?",
                "A week contains how many days?",
                "What is the number of days in a week?",
                "Days per week:",
            ],
            "7"
        ),
        (
            [
                "What color is the sky?",
                "The sky is what color?",
                "What is the color of the sky?",
                "Name the color of the sky.",
            ],
            "blue"
        ),
    ]

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        questions, answer = rng.choice(self.QUESTION_SETS)
        prompt = rng.choice(questions)

        return TestCase(
            id=f"robust_rephrase_{answer[:5]}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=answer,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "rephrase", "answer": answer},
        )


class CaseVariationTemplate(TestTemplate):
    """Same content with different casing."""

    category = "robustness"
    subcategory = "case_variation"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        # Base questions and answers
        qa_pairs = [
            ("What is 2 + 2?", "4"),
            ("What color is grass?", "green"),
            ("How many legs does a dog have?", "4"),
            ("What is the opposite of hot?", "cold"),
            ("What comes after Tuesday?", "Wednesday"),
        ]

        question, answer = rng.choice(qa_pairs)

        # Apply case variation
        if difficulty == Difficulty.EASY:
            # Normal or all lowercase
            if rng.random() < 0.5:
                prompt = question.lower()
            else:
                prompt = question
        elif difficulty == Difficulty.MEDIUM:
            # All uppercase
            prompt = question.upper()
        else:
            # Mixed case
            prompt = "".join(
                c.upper() if rng.random() < 0.5 else c.lower()
                for c in question
            )

        return TestCase(
            id=f"robust_case_{answer}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=answer,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "case_variation"},
        )


class WhitespaceVariationTemplate(TestTemplate):
    """Same content with varying whitespace."""

    category = "robustness"
    subcategory = "whitespace"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        qa_pairs = [
            ("What is 3 + 4?", "7"),
            ("Capital of Japan?", "Tokyo"),
            ("2 times 5 equals", "10"),
            ("Color of snow?", "white"),
        ]

        question, answer = rng.choice(qa_pairs)

        if difficulty == Difficulty.EASY:
            # Normal spacing
            prompt = question
        elif difficulty == Difficulty.MEDIUM:
            # Extra spaces
            prompt = "  ".join(question.split())
        else:
            # Random extra whitespace
            chars = []
            for c in question:
                chars.append(c)
                if c == ' ' and rng.random() < 0.5:
                    chars.append(' ' * rng.randint(1, 3))
            prompt = "".join(chars)

        return TestCase(
            id=f"robust_space_{answer}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=answer,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "whitespace"},
        )


class PunctuationVariationTemplate(TestTemplate):
    """Same content with varying punctuation."""

    category = "robustness"
    subcategory = "punctuation"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        qa_pairs = [
            ("What is the capital of Spain", "Madrid"),
            ("How many hours in a day", "24"),
            ("What planet is closest to the sun", "Mercury"),
            ("What is 10 divided by 2", "5"),
        ]

        base_question, answer = rng.choice(qa_pairs)

        if difficulty == Difficulty.EASY:
            # Normal with question mark
            prompt = base_question + "?"
        elif difficulty == Difficulty.MEDIUM:
            # No punctuation or period
            prompt = rng.choice([base_question, base_question + "."])
        else:
            # Multiple punctuation or unusual
            endings = ["??", "?!", "...", "?...", ":"]
            prompt = base_question + rng.choice(endings)

        return TestCase(
            id=f"robust_punct_{answer}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=answer,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "punctuation"},
        )


class SynonymSubstitutionTemplate(TestTemplate):
    """Question with synonym substitutions."""

    category = "robustness"
    subcategory = "synonym_substitution"

    SUBSTITUTIONS = [
        # (original_phrase, [substitutes], question_template, answer)
        (
            "largest",
            ["biggest", "most enormous", "greatest in size"],
            "What is the {} ocean?",
            "Pacific"
        ),
        (
            "fastest",
            ["quickest", "speediest", "most rapid"],
            "What is the {} land animal?",
            "cheetah"
        ),
        (
            "capital",
            ["capital city", "main city", "governmental seat"],
            "What is the {} of Germany?",
            "Berlin"
        ),
        (
            "sum",
            ["total", "result of adding", "combined value"],
            "What is the {} of 3 and 5?",
            "8"
        ),
    ]

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        original, subs, template, answer = rng.choice(self.SUBSTITUTIONS)

        if difficulty == Difficulty.EASY:
            word = original
        else:
            word = rng.choice(subs)

        prompt = template.format(word)

        return TestCase(
            id=f"robust_synonym_{answer}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=answer,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "synonym_substitution", "original": original},
        )


class OrderVariationTemplate(TestTemplate):
    """Same information, different presentation order."""

    category = "robustness"
    subcategory = "order_variation"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Simple addition - order shouldn't matter
            a, b = rng.randint(1, 20), rng.randint(1, 20)
            if rng.random() < 0.5:
                prompt = f"What is {a} + {b}?"
            else:
                prompt = f"What is {b} + {a}?"
            expected = str(a + b)

        elif difficulty == Difficulty.MEDIUM:
            # Facts in different order
            name = rng.choice(["Alice", "Bob", "Charlie"])
            age = rng.randint(20, 50)
            city = rng.choice(["Paris", "London", "Tokyo"])

            facts = [f"{name} is {age} years old", f"{name} lives in {city}"]
            rng.shuffle(facts)
            question = rng.choice([f"How old is {name}?", f"Where does {name} live?"])

            prompt = ". ".join(facts) + ". " + question
            expected = str(age) if "old" in question else city

        else:
            # Multiplication with multiple terms
            a, b, c = rng.randint(2, 5), rng.randint(2, 5), rng.randint(2, 5)
            terms = [a, b, c]
            rng.shuffle(terms)
            prompt = f"What is {terms[0]} × {terms[1]} × {terms[2]}?"
            expected = str(a * b * c)

        return TestCase(
            id=f"robust_order_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "order_variation"},
        )


# Export all templates
TEMPLATES = [
    RephraseConsistencyTemplate(),
    CaseVariationTemplate(),
    WhitespaceVariationTemplate(),
    PunctuationVariationTemplate(),
    SynonymSubstitutionTemplate(),
    OrderVariationTemplate(),
]
