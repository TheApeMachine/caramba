"""
Reasoning templates.

Tests logical inference abilities including comparison, transitivity,
negation, and conditional reasoning.
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
    random_word,
)


class ComparisonTemplate(TestTemplate):
    """Comparison reasoning: A > B, ask about relationship."""

    category = "reasoning"
    subcategory = "comparison"

    COMPARISONS = [
        ("taller", "shorter", "height"),
        ("older", "younger", "age"),
        ("heavier", "lighter", "weight"),
        ("faster", "slower", "speed"),
        ("larger", "smaller", "size"),
    ]

    NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        comp_more, comp_less, dimension = rng.choice(self.COMPARISONS)

        names = rng.sample(self.NAMES, 2)
        a, b = names[0], names[1]

        if difficulty == Difficulty.EASY:
            # Direct question
            prompt = f"{a} is {comp_more} than {b}. Who is {comp_more}?"
            expected = a
        elif difficulty == Difficulty.MEDIUM:
            # Inverse question
            prompt = f"{a} is {comp_more} than {b}. Who is {comp_less}?"
            expected = b
        else:
            # With distractor
            c = rng.choice([n for n in self.NAMES if n not in names])
            prompt = f"{a} is {comp_more} than {b}. {c} likes pizza. Who is {comp_less}?"
            expected = b

        return TestCase(
            id=f"reason_compare_{a}_{b}_{dimension}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"dimension": dimension, "comparison": comp_more},
        )


class TransitiveTemplate(TestTemplate):
    """Transitive reasoning: A > B, B > C, therefore A > C."""

    category = "reasoning"
    subcategory = "transitive"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        names = rng.sample(["Alice", "Bob", "Charlie", "Diana", "Eve"], 4)

        comparisons = [
            ("taller", "tallest", "shortest"),
            ("older", "oldest", "youngest"),
            ("faster", "fastest", "slowest"),
        ]
        comp, superlative, inverse_sup = rng.choice(comparisons)

        if difficulty == Difficulty.EASY:
            # 2-step chain
            a, b, c = names[:3]
            prompt = f"{a} is {comp} than {b}. {b} is {comp} than {c}. Who is the {superlative}?"
            expected = a
        elif difficulty == Difficulty.MEDIUM:
            # Ask for the least
            a, b, c = names[:3]
            prompt = f"{a} is {comp} than {b}. {b} is {comp} than {c}. Who is the {inverse_sup}?"
            expected = c
        else:
            # 3-step chain
            a, b, c, d = names[:4]
            prompt = f"{a} is {comp} than {b}. {b} is {comp} than {c}. {c} is {comp} than {d}. Who is the {superlative}?"
            expected = a

        return TestCase(
            id=f"reason_transitive_{len(names)}step",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"chain_length": len(names) - 1},
        )


class NegationTemplate(TestTemplate):
    """Negation reasoning: handle NOT statements."""

    category = "reasoning"
    subcategory = "negation"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Simple negation
            facts = [
                ("The sky is blue", "Is the sky blue?", "Yes"),
                ("The sky is not green", "Is the sky green?", "No"),
                ("Dogs bark", "Do dogs bark?", "Yes"),
                ("Cats do not fly", "Do cats fly?", "No"),
            ]
            fact, question, answer = rng.choice(facts)
            prompt = f"{fact}. {question}"
            expected = answer
        elif difficulty == Difficulty.MEDIUM:
            # Double negation
            facts = [
                ("It is not true that dogs cannot bark", "Can dogs bark?", "Yes"),
                ("It is not false that the sun is hot", "Is the sun hot?", "Yes"),
            ]
            fact, question, answer = rng.choice(facts)
            prompt = f"{fact}. {question}"
            expected = answer
        else:
            # Complex negation with context
            prompt = "Alice does not dislike Bob. Bob is not unhappy with Alice. Are Alice and Bob on good terms?"
            expected = "Yes"

        return TestCase(
            id=f"reason_negation_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"negation_depth": difficulty.value},
        )


class ConditionalTemplate(TestTemplate):
    """Conditional reasoning: if-then statements."""

    category = "reasoning"
    subcategory = "conditional"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Modus ponens: If P then Q. P is true. Therefore Q.
            scenarios = [
                ("If it rains, the ground gets wet", "It is raining", "Is the ground wet?", "Yes"),
                ("If you study, you pass", "You studied", "Did you pass?", "Yes"),
            ]
            premise, condition, question, answer = rng.choice(scenarios)
            prompt = f"{premise}. {condition}. {question}"
            expected = answer
        elif difficulty == Difficulty.MEDIUM:
            # Modus tollens: If P then Q. Q is false. Therefore P is false.
            scenarios = [
                ("If it rains, the ground gets wet", "The ground is dry", "Did it rain?", "No"),
                ("If the alarm sounds, there is danger", "There is no danger", "Did the alarm sound?", "No"),
            ]
            premise, condition, question, answer = rng.choice(scenarios)
            prompt = f"{premise}. {condition}. {question}"
            expected = answer
        else:
            # Chained conditionals
            prompt = "If A then B. If B then C. A is true. Is C true?"
            expected = "Yes"

        return TestCase(
            id=f"reason_conditional_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"reasoning_type": "conditional"},
        )


class SetMembershipTemplate(TestTemplate):
    """Set membership reasoning: categories and membership."""

    category = "reasoning"
    subcategory = "set_membership"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Direct membership
            scenarios = [
                ("All dogs are animals", "Fido is a dog", "Is Fido an animal?", "Yes"),
                ("All birds have feathers", "Tweety is a bird", "Does Tweety have feathers?", "Yes"),
            ]
            s1, s2, q, a = rng.choice(scenarios)
            prompt = f"{s1}. {s2}. {q}"
            expected = a
        elif difficulty == Difficulty.MEDIUM:
            # Exclusion
            scenarios = [
                ("No fish can fly", "Nemo is a fish", "Can Nemo fly?", "No"),
                ("No reptiles have fur", "This lizard is a reptile", "Does this lizard have fur?", "No"),
            ]
            s1, s2, q, a = rng.choice(scenarios)
            prompt = f"{s1}. {s2}. {q}"
            expected = a
        else:
            # Chained membership
            prompt = "All mammals are animals. All dogs are mammals. Rex is a dog. Is Rex an animal?"
            expected = "Yes"

        return TestCase(
            id=f"reason_sets_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"reasoning_type": "set_membership"},
        )


class SpatialReasoningTemplate(TestTemplate):
    """Spatial reasoning: positions and relationships."""

    category = "reasoning"
    subcategory = "spatial"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        objects = ["the book", "the pen", "the cup", "the phone", "the lamp"]

        if difficulty == Difficulty.EASY:
            a, b = rng.sample(objects, 2)
            prompt = f"{a.capitalize()} is to the left of {b}. What is to the right of {a}?"
            expected = b.replace("the ", "The ")
        elif difficulty == Difficulty.MEDIUM:
            a, b, c = rng.sample(objects, 3)
            prompt = f"{a.capitalize()} is to the left of {b}. {b.capitalize()} is to the left of {c}. What is in the middle?"
            expected = b.replace("the ", "The ")
        else:
            a, b, c = rng.sample(objects, 3)
            prompt = f"{a.capitalize()} is above {b}. {c.capitalize()} is below {b}. What is between {a} and {c}?"
            expected = b.replace("the ", "The ")

        return TestCase(
            id=f"reason_spatial_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"reasoning_type": "spatial"},
        )


# Export all templates
TEMPLATES = [
    ComparisonTemplate(),
    TransitiveTemplate(),
    NegationTemplate(),
    ConditionalTemplate(),
    SetMembershipTemplate(),
    SpatialReasoningTemplate(),
]
