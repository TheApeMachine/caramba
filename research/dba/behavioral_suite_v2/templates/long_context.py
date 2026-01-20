"""
Long context templates.

Tests ability to retrieve information from various positions in longer contexts,
update tracked values, and maintain focus across extended text.
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
    random_words,
    random_position,
)


class EarlyRetrievalTemplate(TestTemplate):
    """Retrieve information mentioned early in a long context."""

    category = "long_context"
    subcategory = "early_retrieval"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        # Target information
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        colors = ["red", "blue", "green", "yellow", "purple"]
        numbers = [str(rng.randint(10, 99)) for _ in range(5)]

        name = rng.choice(names)
        color = rng.choice(colors)
        number = rng.choice(numbers)

        # Key fact at beginning
        key_fact = f"{name}'s favorite color is {color}."

        # Generate filler
        if difficulty == Difficulty.EASY:
            filler_lines = 3
        elif difficulty == Difficulty.MEDIUM:
            filler_lines = 6
        else:
            filler_lines = 10

        fillers = [
            "The weather was pleasant that day.",
            "Many people were walking in the park.",
            "The store had various items on sale.",
            "Traffic was light in the morning.",
            "Several birds were singing in the trees.",
            "The coffee shop was crowded as usual.",
            "A new restaurant opened downtown.",
            "The library had extended hours this week.",
            "Construction continued on the highway.",
            "The market prices remained stable.",
            "Students were preparing for exams.",
            "The concert tickets sold out quickly.",
        ]
        rng.shuffle(fillers)
        filler_text = " ".join(fillers[:filler_lines])

        # Schema-completion style for base models
        prompt = f"{key_fact} {filler_text}\nName: {name}\nFavorite color:"
        expected = color

        return TestCase(
            id=f"context_early_{name}_{color}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.START,
            metadata={"target_name": name, "filler_lines": filler_lines},
        )


class LateRetrievalTemplate(TestTemplate):
    """Retrieve information mentioned late after distractors."""

    category = "long_context"
    subcategory = "late_retrieval"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        # Generate filler first
        if difficulty == Difficulty.EASY:
            filler_lines = 3
        elif difficulty == Difficulty.MEDIUM:
            filler_lines = 6
        else:
            filler_lines = 10

        fillers = [
            "The morning started with a light drizzle.",
            "Several meetings were scheduled for the afternoon.",
            "The project deadline was moved to next week.",
            "New equipment arrived at the office.",
            "The team lunch was rescheduled.",
            "Annual reports were being prepared.",
            "Client feedback was generally positive.",
            "The server maintenance was completed.",
            "Training sessions were well attended.",
            "Budget reviews were in progress.",
            "The office renovations finished early.",
            "Supply orders were processed quickly.",
        ]
        rng.shuffle(fillers)
        filler_text = " ".join(fillers[:filler_lines])

        # Target information at end
        items = ["laptop", "phone", "book", "keys", "wallet"]
        locations = ["desk", "shelf", "drawer", "bag", "pocket"]
        item = rng.choice(items)
        location = rng.choice(locations)

        key_fact = f"The {item} is in the {location}."
        # Schema-completion style for base models
        prompt = f"{filler_text} {key_fact}\nItem: {item}\nLocation:"
        expected = location

        return TestCase(
            id=f"context_late_{item}_{location}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"target_item": item, "filler_lines": filler_lines},
        )


class MiddleRetrievalTemplate(TestTemplate):
    """Retrieve information from the middle of context."""

    category = "long_context"
    subcategory = "middle_retrieval"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        # Filler before
        if difficulty == Difficulty.EASY:
            filler_before = 2
            filler_after = 2
        elif difficulty == Difficulty.MEDIUM:
            filler_before = 4
            filler_after = 4
        else:
            filler_before = 6
            filler_after = 6

        fillers_a = [
            "The conference room was prepared early.",
            "Coffee was available in the break room.",
            "The presentation slides were updated.",
            "Several emails needed responses.",
            "The printer was working again.",
            "Meeting notes were distributed.",
            "The calendar was synchronized.",
            "Documents were filed properly.",
        ]

        fillers_b = [
            "Lunch orders were collected.",
            "The schedule was confirmed.",
            "Reports were being reviewed.",
            "Invoices were processed.",
            "The system update completed.",
            "Training materials were ready.",
            "Feedback was being gathered.",
            "Progress reports were due.",
        ]

        rng.shuffle(fillers_a)
        rng.shuffle(fillers_b)

        filler_text_before = " ".join(fillers_a[:filler_before])
        filler_text_after = " ".join(fillers_b[:filler_after])

        # Key fact in middle
        name = rng.choice(["Alice", "Bob", "Charlie", "Diana"])
        number = rng.randint(100, 999)
        key_fact = f"{name}'s employee ID is {number}."

        # Schema-completion style for base models
        prompt = f"{filler_text_before} {key_fact} {filler_text_after}\nEmployee: {name}\nID:"
        expected = str(number)

        return TestCase(
            id=f"context_middle_{name}_{number}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.MIDDLE,
            metadata={"target_name": name, "position": "middle"},
        )


class ValueUpdateTemplate(TestTemplate):
    """Track value updates across context."""

    category = "long_context"
    subcategory = "value_update"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        variable = rng.choice(["score", "count", "balance", "total", "points"])

        if difficulty == Difficulty.EASY:
            # Two updates
            initial = rng.randint(10, 50)
            delta1 = rng.randint(5, 15)
            final = initial + delta1
            # Pattern-completion style: show examples then query
            prompt = f"start=10, +5 -> final=15\nstart=20, +10 -> final=30\nstart={initial}, +{delta1} -> final="
            expected = str(final)

        elif difficulty == Difficulty.MEDIUM:
            # Three updates with mixed operations
            initial = rng.randint(50, 100)
            delta1 = rng.randint(10, 20)
            delta2 = rng.randint(5, 15)
            final = initial + delta1 - delta2
            prompt = f"The {variable} starts at {initial}. It increases by {delta1}, then decreases by {delta2}. What is the final {variable}?"
            expected = str(final)

        else:
            # Multiple updates with distractors
            initial = rng.randint(100, 200)
            delta1 = rng.randint(20, 40)
            delta2 = rng.randint(10, 30)
            delta3 = rng.randint(15, 25)
            final = initial + delta1 - delta2 + delta3

            distractor = f"The weather was sunny. Many people were outside."
            prompt = f"The {variable} starts at {initial}. It increases by {delta1}. {distractor} Then it decreases by {delta2}. Finally it increases by {delta3}. What is the final {variable}?"
            expected = str(final)

        return TestCase(
            id=f"context_update_{variable}_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"variable": variable, "type": "value_update"},
        )


class MultiFactRetrievalTemplate(TestTemplate):
    """Retrieve multiple related facts from context."""

    category = "long_context"
    subcategory = "multi_fact"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        names = ["Alice", "Bob", "Charlie", "Diana"]
        cities = ["Paris", "London", "Tokyo", "Berlin"]
        jobs = ["engineer", "teacher", "doctor", "artist"]

        # Assign facts
        rng.shuffle(names)
        rng.shuffle(cities)
        rng.shuffle(jobs)

        facts = []
        for i, name in enumerate(names[:3]):
            facts.append(f"{name} lives in {cities[i]}.")
            facts.append(f"{name} works as a {jobs[i]}.")

        rng.shuffle(facts)

        if difficulty == Difficulty.EASY:
            # Single fact question (schema-style)
            target_name = names[0]
            prompt = " ".join(facts) + f"\nPerson: {target_name}\nCity:"
            expected = cities[0]
        elif difficulty == Difficulty.MEDIUM:
            # Job question (schema-style)
            target_name = names[1]
            prompt = " ".join(facts) + f"\nPerson: {target_name}\nJob:"
            expected = jobs[1]
        else:
            # Cross-reference question (schema-style)
            target_city = cities[0]
            prompt = " ".join(facts) + f"\nCity: {target_city}\nResident:"
            expected = names[0]

        return TestCase(
            id=f"context_multi_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=random_position(rng),
            metadata={"type": "multi_fact"},
        )


class InstructionFirstTemplate(TestTemplate):
    """Follow an instruction given at the start of context."""

    category = "long_context"
    subcategory = "instruction_first"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Few-shot first extraction
            pool1 = rng.sample(["apple", "banana", "cherry", "date"], 3)
            pool2 = rng.sample(["dog", "cat", "bird", "fish"], 3)
            pool3 = rng.sample(["red", "blue", "green", "yellow"], 3)
            prompt = (
                f"List: {', '.join(pool1)}\nFirst: {pool1[0]}\n"
                f"List: {', '.join(pool2)}\nFirst: {pool2[0]}\n"
                f"List: {', '.join(pool3)}\nFirst:"
            )
            expected = pool3[0]
        elif difficulty == Difficulty.MEDIUM:
            # Few-shot last extraction with filler
            animals = rng.sample(["elephant", "giraffe", "hippo", "zebra", "lion"], 4)
            filler = "Animals: " + ", ".join(animals) + ". These are zoo animals."
            pool1 = rng.sample(["north", "south", "east", "west"], 3)
            pool2 = rng.sample(["gold", "silver", "bronze"], 3)
            prompt = (
                f"List: {', '.join(pool1)}\nLast: {pool1[-1]}\n"
                f"{filler}\n"
                f"List: {', '.join(pool2)}\nLast:"
            )
            expected = pool2[-1]
        else:
            # Few-shot sum calculation
            nums1 = [rng.randint(10, 30) for _ in range(3)]
            nums2 = [rng.randint(10, 30) for _ in range(3)]
            nums3 = [rng.randint(10, 30) for _ in range(4)]
            prompt = (
                f"{', '.join(map(str, nums1))} -> sum={sum(nums1)}\n"
                f"{', '.join(map(str, nums2))} -> sum={sum(nums2)}\n"
                f"{', '.join(map(str, nums3))} -> sum="
            )
            expected = str(sum(nums3))

        return TestCase(
            id=f"context_pattern_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.START,
            metadata={"type": "pattern_first"},
        )


# Export all templates
TEMPLATES = [
    EarlyRetrievalTemplate(),
    LateRetrievalTemplate(),
    MiddleRetrievalTemplate(),
    ValueUpdateTemplate(),
    MultiFactRetrievalTemplate(),
    InstructionFirstTemplate(),
]
