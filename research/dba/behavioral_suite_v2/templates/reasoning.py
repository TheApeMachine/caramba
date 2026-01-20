"""
Reasoning templates.

Tests logical inference abilities including comparison, transitivity,
negation, and conditional reasoning.

NOTE: These templates use CHOICE_LOGPROB scoring for clean capability measurement,
removing decoder/termination confounds. The choices field contains the candidate
answers that will be scored via log-probability.
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
            # Few-shot pattern: fact -> label
            prompt = f"Eve is older than Dan. Older: Eve\n{a} is {comp_more} than {b}. {comp_more.capitalize()}:"
            expected = f" {a}"
            choices = [f" {a}", f" {b}"]
        elif difficulty == Difficulty.MEDIUM:
            # Inverse question with few-shot
            prompt = f"Eve is older than Dan. Younger: Dan\n{a} is {comp_more} than {b}. {comp_less.capitalize()}:"
            expected = f" {b}"
            choices = [f" {a}", f" {b}"]
        else:
            # With distractor
            c = rng.choice([n for n in self.NAMES if n not in names])
            prompt = f"Eve is older than Dan. Younger: Dan\n{a} is {comp_more} than {b}. {c} likes pizza. {comp_less.capitalize()}:"
            expected = f" {b}"
            choices = [f" {a}", f" {b}", f" {c}"]

        return TestCase(
            id=f"reason_compare_{a}_{b}_{dimension}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
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
            # 2-step chain with few-shot example
            a, b, c = names[:3]
            prompt = f"X > Y, Y > Z. Tallest: X\n{a} is {comp} than {b}. {b} is {comp} than {c}. {superlative.capitalize()}:"
            expected = f" {a}"
            choices = [f" {a}", f" {b}", f" {c}"]
        elif difficulty == Difficulty.MEDIUM:
            # Ask for the least with few-shot
            a, b, c = names[:3]
            prompt = f"X > Y, Y > Z. Smallest: Z\n{a} is {comp} than {b}. {b} is {comp} than {c}. {inverse_sup.capitalize()}:"
            expected = f" {c}"
            choices = [f" {a}", f" {b}", f" {c}"]
        else:
            # 3-step chain
            a, b, c, d = names[:4]
            prompt = f"X > Y, Y > Z. Tallest: X\n{a} is {comp} than {b}. {b} is {comp} than {c}. {c} is {comp} than {d}. {superlative.capitalize()}:"
            expected = f" {a}"
            choices = [f" {a}", f" {b}", f" {c}", f" {d}"]

        return TestCase(
            id=f"reason_transitive_{len(names)}step",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"chain_length": len(names) - 1},
        )


class NegationTemplate(TestTemplate):
    """Negation reasoning: handle NOT statements."""

    category = "reasoning"
    subcategory = "negation"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        # Standard yes/no choices for all negation tests
        choices = [" yes", " no"]

        if difficulty == Difficulty.EASY:
            # Simple negation with few-shot (Q: A: format)
            facts = [
                ("The sky is blue", "Is the sky blue?", " yes"),
                ("The sky is not green", "Is the sky green?", " no"),
                ("Dogs bark", "Do dogs bark?", " yes"),
                ("Cats do not fly", "Do cats fly?", " no"),
            ]
            fact, question, expected = rng.choice(facts)
            prompt = f"Q: Is 1+1=2? A: yes\nQ: Is 1+1=3? A: no\n{fact}. Q: {question} A:"
        elif difficulty == Difficulty.MEDIUM:
            # Double negation with few-shot
            facts = [
                ("It is not true that dogs cannot bark", "bark?", " yes"),
                ("It is not false that the sun is hot", "hot?", " yes"),
            ]
            fact, question, expected = rng.choice(facts)
            prompt = f"Q: Can dogs bark? A: yes\n{fact}. Q: {question} A:"
        else:
            # Complex negation
            prompt = "Q: Are A and B friends? A: yes\nAlice does not dislike Bob. Bob is not unhappy with Alice. Q: Are Alice and Bob on good terms? A:"
            expected = " yes"

        return TestCase(
            id=f"reason_negation_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"negation_depth": difficulty.value},
        )


class ConditionalTemplate(TestTemplate):
    """Conditional reasoning: if-then statements."""

    category = "reasoning"
    subcategory = "conditional"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        # Standard Yes/No choices
        choices = [" Yes", " No"]

        if difficulty == Difficulty.EASY:
            # Modus ponens: If P then Q. P is true. Therefore Q.
            scenarios = [
                ("If it rains, the ground gets wet", "It is raining", "Is the ground wet?", " Yes"),
                ("If you study, you pass", "You studied", "Did you pass?", " Yes"),
            ]
            premise, condition, question, expected = rng.choice(scenarios)
            prompt = f"{premise}. {condition}. {question}"
        elif difficulty == Difficulty.MEDIUM:
            # Modus tollens: If P then Q. Q is false. Therefore P is false.
            scenarios = [
                ("If it rains, the ground gets wet", "The ground is dry", "Did it rain?", " No"),
                ("If the alarm sounds, there is danger", "There is no danger", "Did the alarm sound?", " No"),
            ]
            premise, condition, question, expected = rng.choice(scenarios)
            prompt = f"{premise}. {condition}. {question}"
        else:
            # Chained conditionals
            prompt = "If A then B. If B then C. A is true. Is C true?"
            expected = " Yes"

        return TestCase(
            id=f"reason_conditional_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"reasoning_type": "conditional"},
        )


class SetMembershipTemplate(TestTemplate):
    """Set membership reasoning: categories and membership."""

    category = "reasoning"
    subcategory = "set_membership"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        # Standard Yes/No choices
        choices = [" Yes", " No"]

        if difficulty == Difficulty.EASY:
            # Direct membership
            scenarios = [
                ("All dogs are animals", "Fido is a dog", "Is Fido an animal?", " Yes"),
                ("All birds have feathers", "Tweety is a bird", "Does Tweety have feathers?", " Yes"),
            ]
            s1, s2, q, expected = rng.choice(scenarios)
            prompt = f"{s1}. {s2}. {q}"
        elif difficulty == Difficulty.MEDIUM:
            # Exclusion
            scenarios = [
                ("No fish can fly", "Nemo is a fish", "Can Nemo fly?", " No"),
                ("No reptiles have fur", "This lizard is a reptile", "Does this lizard have fur?", " No"),
            ]
            s1, s2, q, expected = rng.choice(scenarios)
            prompt = f"{s1}. {s2}. {q}"
        else:
            # Chained membership
            prompt = "All mammals are animals. All dogs are mammals. Rex is a dog. Is Rex an animal?"
            expected = " Yes"

        return TestCase(
            id=f"reason_sets_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
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
            expected = " " + b.replace("the ", "The ")
            # Choices are the two objects
            choices = [" " + a.replace("the ", "The "), " " + b.replace("the ", "The ")]
        elif difficulty == Difficulty.MEDIUM:
            a, b, c = rng.sample(objects, 3)
            prompt = f"{a.capitalize()} is to the left of {b}. {b.capitalize()} is to the left of {c}. What is in the middle?"
            expected = " " + b.replace("the ", "The ")
            choices = [" " + x.replace("the ", "The ") for x in [a, b, c]]
        else:
            a, b, c = rng.sample(objects, 3)
            prompt = f"{a.capitalize()} is above {b}. {c.capitalize()} is below {b}. What is between {a} and {c}?"
            expected = " " + b.replace("the ", "The ")
            choices = [" " + x.replace("the ", "The ") for x in [a, b, c]]

        return TestCase(
            id=f"reason_spatial_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
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
