"""
Instruction following templates.

Tests ability to follow explicit instructions about output format,
content selection, and task constraints.
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


class ReturnFirstTemplate(TestTemplate):
    """Return only the first item from a list."""

    category = "instruction_following"
    subcategory = "return_first"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            items = rng.sample(["apple", "banana", "cherry", "date"], 3)
        elif difficulty == Difficulty.MEDIUM:
            items = rng.sample(["red", "blue", "green", "yellow", "purple"], 4)
        else:
            items = [str(rng.randint(10, 99)) for _ in range(5)]

        prompt = f"Return only the first item from this list: {', '.join(items)}"
        expected = items[0]

        return TestCase(
            id=f"instruct_first_{expected[:5]}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.START,
            metadata={"type": "return_first"},
        )


class ReturnLastTemplate(TestTemplate):
    """Return only the last item from a list."""

    category = "instruction_following"
    subcategory = "return_last"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            items = rng.sample(["dog", "cat", "bird", "fish"], 3)
        elif difficulty == Difficulty.MEDIUM:
            items = rng.sample(["Paris", "London", "Tokyo", "Berlin", "Rome"], 4)
        else:
            items = [str(rng.randint(100, 999)) for _ in range(6)]

        prompt = f"Return only the last item from this list: {', '.join(items)}"
        expected = items[-1]

        return TestCase(
            id=f"instruct_last_{expected[:5]}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "return_last"},
        )


class CountWordsTemplate(TestTemplate):
    """Count the number of words."""

    category = "instruction_following"
    subcategory = "count_words"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        word_pool = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                     "big", "small", "red", "blue", "happy", "sad", "old", "new"]

        if difficulty == Difficulty.EASY:
            num_words = rng.randint(3, 5)
        elif difficulty == Difficulty.MEDIUM:
            num_words = rng.randint(5, 8)
        else:
            num_words = rng.randint(8, 12)

        words = rng.sample(word_pool, min(num_words, len(word_pool)))
        if len(words) < num_words:
            words = words + rng.choices(word_pool, k=num_words - len(words))

        sentence = " ".join(words)
        prompt = f"Count the number of words in this sentence: \"{sentence}\""
        expected = str(len(words))

        return TestCase(
            id=f"instruct_count_{num_words}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"word_count": num_words, "type": "count_words"},
        )


class UppercaseTemplate(TestTemplate):
    """Convert text to uppercase."""

    category = "instruction_following"
    subcategory = "uppercase"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            word = rng.choice(["hello", "world", "test", "data"])
        elif difficulty == Difficulty.MEDIUM:
            words = rng.sample(["quick", "brown", "fox", "jumps"], 2)
            word = " ".join(words)
        else:
            words = rng.sample(["the", "lazy", "dog", "sleeps", "today"], 3)
            word = " ".join(words)

        prompt = f"Convert to uppercase: {word}"
        expected = word.upper()

        return TestCase(
            id=f"instruct_upper_{word[:5]}",
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
    """Reverse a string or list."""

    category = "instruction_following"
    subcategory = "reverse"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Short word
            word = rng.choice(["cat", "dog", "sun", "hat"])
            prompt = f"Reverse this word: {word}"
            expected = word[::-1]
        elif difficulty == Difficulty.MEDIUM:
            # Longer word
            word = rng.choice(["hello", "world", "python", "banana"])
            prompt = f"Reverse this word: {word}"
            expected = word[::-1]
        else:
            # List of items
            items = rng.sample(["A", "B", "C", "D", "E"], 4)
            prompt = f"Reverse this list: {', '.join(items)}"
            expected = ", ".join(items[::-1])

        return TestCase(
            id=f"instruct_reverse_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "reverse"},
        )


class FilterByConditionTemplate(TestTemplate):
    """Filter items by a condition."""

    category = "instruction_following"
    subcategory = "filter"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Filter numbers > threshold
            numbers = [rng.randint(1, 20) for _ in range(5)]
            threshold = 10
            prompt = f"List only the numbers greater than {threshold}: {', '.join(map(str, numbers))}"
            filtered = [n for n in numbers if n > threshold]
            expected = ", ".join(map(str, filtered)) if filtered else "none"

        elif difficulty == Difficulty.MEDIUM:
            # Filter by starting letter
            words = rng.sample(["apple", "banana", "apricot", "cherry", "avocado", "berry"], 5)
            prompt = f"List only words starting with 'a': {', '.join(words)}"
            filtered = [w for w in words if w.startswith('a')]
            expected = ", ".join(filtered) if filtered else "none"

        else:
            # Filter even numbers
            numbers = [rng.randint(1, 50) for _ in range(6)]
            prompt = f"List only the even numbers: {', '.join(map(str, numbers))}"
            filtered = [n for n in numbers if n % 2 == 0]
            expected = ", ".join(map(str, filtered)) if filtered else "none"

        return TestCase(
            id=f"instruct_filter_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "filter"},
        )


class OutputFormatTemplate(TestTemplate):
    """Follow specific output format instructions."""

    category = "instruction_following"
    subcategory = "output_format"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Answer with yes/no only
            questions = [
                ("Is 2 + 2 equal to 4?", "yes"),
                ("Is the sky green?", "no"),
                ("Do dogs bark?", "yes"),
                ("Can fish fly?", "no"),
            ]
            question, answer = rng.choice(questions)
            prompt = f"Answer only 'yes' or 'no': {question}"
            expected = answer

        elif difficulty == Difficulty.MEDIUM:
            # Answer with single word
            facts = [
                ("The capital of France is what city? Answer with one word only.", "Paris"),
                ("What color is grass? Answer with one word only.", "green"),
                ("What is 5 + 3? Answer with just the number.", "8"),
            ]
            prompt, expected = rng.choice(facts)

        else:
            # Structured format
            name = rng.choice(["Alice", "Bob", "Charlie"])
            age = rng.randint(20, 40)
            prompt = f"Output ONLY the age as a number, nothing else. {name} is {age} years old."
            expected = str(age)

        return TestCase(
            id=f"instruct_format_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "output_format"},
        )


class SelectiveExtractionTemplate(TestTemplate):
    """Extract only specific information as instructed."""

    category = "instruction_following"
    subcategory = "selective_extraction"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Extract name only
            name = rng.choice(["Alice", "Bob", "Charlie"])
            age = rng.randint(20, 50)
            city = rng.choice(["Paris", "London", "Tokyo"])
            prompt = f"Extract only the name: {name} is {age} years old and lives in {city}."
            expected = name

        elif difficulty == Difficulty.MEDIUM:
            # Extract number only
            item = rng.choice(["apples", "oranges", "books"])
            count = rng.randint(5, 20)
            prompt = f"Extract only the number: There are {count} {item} on the table."
            expected = str(count)

        else:
            # Extract specific field from structured data
            data = {
                "name": rng.choice(["Alice", "Bob"]),
                "age": rng.randint(20, 40),
                "city": rng.choice(["Paris", "London"]),
            }
            field = rng.choice(["name", "city"])
            prompt = f"Extract only the '{field}' value: name={data['name']}, age={data['age']}, city={data['city']}"
            expected = str(data[field])

        return TestCase(
            id=f"instruct_extract_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "selective_extraction"},
        )


# Export all templates
TEMPLATES = [
    ReturnFirstTemplate(),
    ReturnLastTemplate(),
    CountWordsTemplate(),
    UppercaseTemplate(),
    ReverseTemplate(),
    FilterByConditionTemplate(),
    OutputFormatTemplate(),
    SelectiveExtractionTemplate(),
]
