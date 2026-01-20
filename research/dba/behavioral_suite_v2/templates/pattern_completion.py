"""
Pattern completion templates.

Tests ability to complete patterns, extract elements, and apply transformations
using continuation-native prompts (few-shot completion style).

Designed for base (non-instruction-tuned) models where the model completes
the next line in a clear template pattern.
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


class ExtractFirstTemplate(TestTemplate):
    """Extract the first item from a list via pattern completion."""

    category = "pattern_completion"
    subcategory = "extract_first"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        # Generate item pools based on difficulty
        if difficulty == Difficulty.EASY:
            pool1 = rng.sample(["apple", "banana", "cherry", "date"], 3)
            pool2 = rng.sample(["dog", "cat", "bird", "fish"], 3)
            pool3 = rng.sample(["red", "blue", "green", "yellow"], 3)
        elif difficulty == Difficulty.MEDIUM:
            pool1 = rng.sample(["elephant", "giraffe", "zebra", "lion"], 3)
            pool2 = rng.sample(["Paris", "London", "Tokyo", "Rome"], 3)
            pool3 = rng.sample(["Monday", "Tuesday", "Friday", "Sunday"], 3)
        else:
            pool1 = [str(rng.randint(10, 99)) for _ in range(4)]
            pool2 = [str(rng.randint(100, 999)) for _ in range(4)]
            pool3 = [str(rng.randint(10, 99)) for _ in range(4)]

        # Few-shot examples + query
        lines = [
            f"Items: {', '.join(pool1)}",
            f"First: {pool1[0]}",
            f"Items: {', '.join(pool2)}",
            f"First: {pool2[0]}",
            f"Items: {', '.join(pool3)}",
            "First:",
        ]
        prompt = "\n".join(lines)
        expected = pool3[0]

        return TestCase(
            id=f"pattern_first_{expected[:5]}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.START,
            metadata={"type": "extract_first"},
        )


class ExtractLastTemplate(TestTemplate):
    """Extract the last item from a list via pattern completion."""

    category = "pattern_completion"
    subcategory = "extract_last"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            pool1 = rng.sample(["one", "two", "three"], 3)
            pool2 = rng.sample(["alpha", "beta", "gamma"], 3)
            pool3 = rng.sample(["sun", "moon", "star"], 3)
        elif difficulty == Difficulty.MEDIUM:
            pool1 = rng.sample(["spring", "summer", "fall", "winter"], 3)
            pool2 = rng.sample(["north", "south", "east", "west"], 3)
            pool3 = rng.sample(["gold", "silver", "bronze", "copper"], 3)
        else:
            pool1 = [str(rng.randint(10, 99)) for _ in range(4)]
            pool2 = [str(rng.randint(10, 99)) for _ in range(4)]
            pool3 = [str(rng.randint(10, 99)) for _ in range(5)]

        lines = [
            f"Items: {', '.join(pool1)}",
            f"Last: {pool1[-1]}",
            f"Items: {', '.join(pool2)}",
            f"Last: {pool2[-1]}",
            f"Items: {', '.join(pool3)}",
            "Last:",
        ]
        prompt = "\n".join(lines)
        expected = pool3[-1]

        return TestCase(
            id=f"pattern_last_{expected[:5]}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "extract_last"},
        )


class CountWordsTemplate(TestTemplate):
    """Count words via pattern completion."""

    category = "pattern_completion"
    subcategory = "count_words"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        word_pool = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                     "big", "small", "red", "blue", "happy", "sad", "old", "new"]

        # Generate example sentences
        if difficulty == Difficulty.EASY:
            lens = [3, 4, 3]
        elif difficulty == Difficulty.MEDIUM:
            lens = [4, 5, 6]
        else:
            lens = [5, 7, 9]

        sentences = []
        for l in lens:
            words = rng.sample(word_pool, min(l, len(word_pool)))
            if len(words) < l:
                words = words + rng.choices(word_pool, k=l - len(words))
            sentences.append(" ".join(words))

        lines = [
            f'"{sentences[0]}" -> {lens[0]}',
            f'"{sentences[1]}" -> {lens[1]}',
            f'"{sentences[2]}" ->',
        ]
        prompt = "\n".join(lines)
        expected = str(lens[2])

        return TestCase(
            id=f"pattern_count_{lens[2]}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"word_count": lens[2], "type": "count_words"},
        )


class UppercaseTemplate(TestTemplate):
    """Convert to uppercase via pattern completion."""

    category = "pattern_completion"
    subcategory = "uppercase"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            examples = [("cat", "CAT"), ("dog", "DOG")]
            query = rng.choice(["sun", "hat", "box"])
        elif difficulty == Difficulty.MEDIUM:
            examples = [("hello", "HELLO"), ("world", "WORLD")]
            query = rng.choice(["apple", "table", "chair"])
        else:
            examples = [("quick brown", "QUICK BROWN"), ("lazy dog", "LAZY DOG")]
            words = rng.sample(["red", "blue", "big", "small"], 2)
            query = " ".join(words)

        lines = []
        for lower, upper in examples:
            lines.append(f"{lower} -> {upper}")
        lines.append(f"{query} ->")
        
        prompt = "\n".join(lines)
        expected = query.upper()

        return TestCase(
            id=f"pattern_upper_{query[:5]}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "uppercase"},
        )


class ReverseTemplate(TestTemplate):
    """Reverse a string via pattern completion."""

    category = "pattern_completion"
    subcategory = "reverse"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            examples = [("abc", "cba"), ("dog", "god")]
            query = rng.choice(["cat", "top", "rat"])
        elif difficulty == Difficulty.MEDIUM:
            examples = [("hello", "olleh"), ("world", "dlrow")]
            query = rng.choice(["star", "moon", "door"])
        else:
            examples = [("A, B, C", "C, B, A"), ("1, 2, 3, 4", "4, 3, 2, 1")]
            items = rng.sample(["X", "Y", "Z", "W"], 3)
            query = ", ".join(items)

        lines = []
        for original, reversed_val in examples:
            lines.append(f"{original} -> {reversed_val}")
        lines.append(f"{query} ->")
        
        prompt = "\n".join(lines)
        
        if difficulty == Difficulty.HARD:
            expected = ", ".join(query.split(", ")[::-1])
        else:
            expected = query[::-1]

        return TestCase(
            id=f"pattern_reverse_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "reverse"},
        )


class FilterGreaterTemplate(TestTemplate):
    """Filter numbers greater than threshold via pattern completion."""

    category = "pattern_completion"
    subcategory = "filter_greater"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        threshold = 10

        # Generate example sets
        set1 = [rng.randint(1, 20) for _ in range(4)]
        set2 = [rng.randint(1, 20) for _ in range(4)]
        set3 = [rng.randint(1, 20) for _ in range(5 if difficulty != Difficulty.EASY else 4)]

        filter1 = [n for n in set1 if n > threshold]
        filter2 = [n for n in set2 if n > threshold]
        filter3 = [n for n in set3 if n > threshold]

        def fmt_list(lst):
            return ", ".join(map(str, lst)) if lst else "none"

        lines = [
            f"Numbers: {', '.join(map(str, set1))} | >{threshold}: {fmt_list(filter1)}",
            f"Numbers: {', '.join(map(str, set2))} | >{threshold}: {fmt_list(filter2)}",
            f"Numbers: {', '.join(map(str, set3))} | >{threshold}:",
        ]
        prompt = "\n".join(lines)
        expected = fmt_list(filter3)

        return TestCase(
            id=f"pattern_filter_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"threshold": threshold, "type": "filter"},
        )


class YesNoPatternTemplate(TestTemplate):
    """Yes/no questions via pattern completion (Q: A: format)."""

    category = "pattern_completion"
    subcategory = "yes_no"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        qa_pairs = [
            ("Is 1 + 1 equal to 2?", "yes"),
            ("Is the sky blue?", "yes"),
            ("Do dogs bark?", "yes"),
            ("Is 2 + 2 equal to 5?", "no"),
            ("Is the sun cold?", "no"),
            ("Can fish fly?", "no"),
            ("Do cats meow?", "yes"),
            ("Is water dry?", "no"),
        ]

        rng.shuffle(qa_pairs)
        examples = qa_pairs[:2]
        query_q, query_a = qa_pairs[2]

        lines = []
        for q, a in examples:
            lines.append(f"Q: {q} A: {a}")
        lines.append(f"Q: {query_q} A:")

        prompt = "\n".join(lines)
        expected = query_a

        return TestCase(
            id=f"pattern_yesno_{expected}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "yes_no"},
        )


class ExtractFieldTemplate(TestTemplate):
    """Extract a specific field from structured data via pattern completion."""

    category = "pattern_completion"
    subcategory = "extract_field"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        names = ["Alice", "Bob", "Charlie", "Diana"]
        cities = ["Paris", "London", "Tokyo", "Rome"]
        ages = [25, 30, 35, 40]

        rng.shuffle(names)
        rng.shuffle(cities)
        rng.shuffle(ages)

        if difficulty == Difficulty.EASY:
            # Extract name
            lines = [
                f"name={names[0]}, city={cities[0]} -> name: {names[0]}",
                f"name={names[1]}, city={cities[1]} -> name: {names[1]}",
                f"name={names[2]}, city={cities[2]} -> name:",
            ]
            expected = names[2]
        elif difficulty == Difficulty.MEDIUM:
            # Extract city
            lines = [
                f"name={names[0]}, city={cities[0]} -> city: {cities[0]}",
                f"name={names[1]}, city={cities[1]} -> city: {cities[1]}",
                f"name={names[2]}, city={cities[2]} -> city:",
            ]
            expected = cities[2]
        else:
            # Extract age (numbers)
            lines = [
                f"person={names[0]}, age={ages[0]} -> age: {ages[0]}",
                f"person={names[1]}, age={ages[1]} -> age: {ages[1]}",
                f"person={names[2]}, age={ages[2]} -> age:",
            ]
            expected = str(ages[2])

        prompt = "\n".join(lines)

        return TestCase(
            id=f"pattern_extract_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "extract_field"},
        )


# Export all templates
TEMPLATES = [
    ExtractFirstTemplate(),
    ExtractLastTemplate(),
    CountWordsTemplate(),
    UppercaseTemplate(),
    ReverseTemplate(),
    FilterGreaterTemplate(),
    YesNoPatternTemplate(),
    ExtractFieldTemplate(),
]
