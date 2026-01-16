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

        prompt = f"{key_fact} {filler_text} What is {name}'s favorite color?"
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
        prompt = f"{filler_text} {key_fact} Where is the {item}?"
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

        prompt = f"{filler_text_before} {key_fact} {filler_text_after} What is {name}'s employee ID?"
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
            prompt = f"The {variable} starts at {initial}. Then it increases by {delta1}. What is the final {variable}?"
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
            # Single fact question
            target_name = names[0]
            prompt = " ".join(facts) + f" Where does {target_name} live?"
            expected = cities[0]
        elif difficulty == Difficulty.MEDIUM:
            # Job question
            target_name = names[1]
            prompt = " ".join(facts) + f" What is {target_name}'s job?"
            expected = jobs[1]
        else:
            # Cross-reference question
            target_city = cities[0]
            prompt = " ".join(facts) + f" Who lives in {target_city}?"
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
            words = rng.sample(["apple", "banana", "cherry", "date"], 3)
            instruction = "Return only the first word from the list below."
            content = ", ".join(words)
            expected = words[0]
        elif difficulty == Difficulty.MEDIUM:
            words = rng.sample(["elephant", "giraffe", "hippo", "zebra", "lion"], 4)
            instruction = "Return only the last word from the list below."
            filler = "This is a list of animals commonly found in zoos."
            content = f"{filler} {', '.join(words)}"
            expected = words[-1]
        else:
            numbers = [str(rng.randint(10, 99)) for _ in range(5)]
            instruction = "Return the sum of all numbers in the list below."
            filler = "Here are some randomly generated numbers for analysis."
            content = f"{filler} {', '.join(numbers)}"
            expected = str(sum(int(n) for n in numbers))

        prompt = f"{instruction}\n\n{content}"

        return TestCase(
            id=f"context_instruct_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.START,
            metadata={"type": "instruction_first"},
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
