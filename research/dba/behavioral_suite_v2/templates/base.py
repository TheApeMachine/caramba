"""
Base template classes for behavioral test generation.

Templates define parameterized test structures that can be instantiated
with randomized values to produce many test variants.
"""
from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Literal


class Difficulty(Enum):
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()


class TargetPosition(Enum):
    START = auto()
    MIDDLE = auto()
    END = auto()


class EvalKind(Enum):
    """Evaluation method for a test case."""
    GENERATION = "generation"  # Free-form generation, evaluated by scoring.py
    EXACT_MATCH_GREEDY = "exact_match_greedy"
    CHOICE_LOGPROB = "choice_logprob"
    INT_GREEDY = "int_greedy"
    FLOAT_GREEDY = "float_greedy"
    CONTAINS = "contains"
    PREFIX = "prefix"


@dataclass
class TestCase:
    """A single behavioral test case."""
    id: str
    category: str
    subcategory: str
    difficulty: Difficulty

    prompt: str
    expected: str | int | float

    kind: EvalKind
    match: str = "first_line"  # For exact_match: first_line, full, any_line
    choices: list[str] | None = None  # For choice_logprob

    # Metadata
    target_position: TargetPosition | None = None
    template_id: str = ""
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional template-specific data

    # For attention analysis
    critical_tokens: list[int] | None = None  # Token indices model should attend to
    distractor_tokens: list[int] | None = None  # Token indices to ignore


@dataclass
class TemplateSlot:
    """A slot in a template that can be filled with values."""
    name: str
    values: list[Any] | Callable[[], Any]
    constraints: dict[str, Any] = field(default_factory=dict)


class TestTemplate(ABC):
    """Abstract base class for test templates."""

    category: str
    subcategory: str
    template_id: str
    difficulty: Difficulty

    @abstractmethod
    def generate(self, rng: random.Random) -> TestCase:
        """Generate a test case from this template."""
        pass

    def generate_batch(
        self,
        count: int,
        seed: int = 42,
    ) -> list[TestCase]:
        """Generate multiple test cases with different seeds."""
        cases = []
        for i in range(count):
            rng = random.Random(seed + i)
            case = self.generate(rng)
            case.seed = seed + i
            case.template_id = self.template_id
            cases.append(case)
        return cases


# =============================================================================
# Utility Functions for Template Generation
# =============================================================================

def random_word(rng: random.Random, length: int = 5) -> str:
    """Generate a random lowercase word."""
    return ''.join(rng.choices(string.ascii_lowercase, k=length))


def random_upper_word(rng: random.Random, length: int = 5) -> str:
    """Generate a random uppercase word."""
    return ''.join(rng.choices(string.ascii_uppercase, k=length))


def random_alphanumeric(rng: random.Random, length: int = 6) -> str:
    """Generate random alphanumeric string."""
    return ''.join(rng.choices(string.ascii_uppercase + string.digits, k=length))


def random_number(rng: random.Random, min_val: int, max_val: int) -> int:
    """Generate random integer in range."""
    return rng.randint(min_val, max_val)


def random_sequence(
    rng: random.Random,
    length: int,
    item_generator: Callable[[random.Random], str],
) -> list[str]:
    """Generate a sequence of random items."""
    return [item_generator(rng) for _ in range(length)]


def shuffle_with_target_position(
    rng: random.Random,
    items: list[Any],
    target_idx: int,
    position: TargetPosition,
) -> tuple[list[Any], int]:
    """
    Shuffle items, placing target at specified position.

    Returns:
        Tuple of (shuffled_items, new_target_index)
    """
    target = items[target_idx]
    others = items[:target_idx] + items[target_idx + 1:]
    rng.shuffle(others)

    if position == TargetPosition.START:
        new_idx = 0
        result = [target] + others
    elif position == TargetPosition.END:
        new_idx = len(items) - 1
        result = others + [target]
    else:  # MIDDLE
        new_idx = len(others) // 2
        result = others[:new_idx] + [target] + others[new_idx:]

    return result, new_idx


def create_fewshot_prompt(
    examples: list[tuple[str, str]],
    query: str,
    separator: str = " -> ",
    line_separator: str = "\n",
) -> str:
    """Create a few-shot prompt from examples and query."""
    lines = [f"{inp}{separator}{out}" for inp, out in examples]
    lines.append(f"{query}{separator}")
    return line_separator.join(lines)


def create_distractor_prompt(
    target: str,
    distractors: list[str],
    rng: random.Random,
    target_position: TargetPosition = TargetPosition.END,
) -> tuple[str, int]:
    """
    Create a prompt with target and distractors.

    Returns:
        Tuple of (prompt, target_line_index)
    """
    all_items = distractors + [target]
    rng.shuffle(all_items[:len(distractors)])  # Shuffle distractors only

    # Place target at specified position
    if target_position == TargetPosition.START:
        result = [target] + all_items[:-1]
        target_idx = 0
    elif target_position == TargetPosition.END:
        result = all_items
        target_idx = len(result) - 1
    else:
        mid = len(all_items) // 2
        result = all_items[:mid] + [target] + all_items[mid:-1]
        target_idx = mid

    return result, target_idx


# =============================================================================
# Common Word/Data Banks
# =============================================================================

COMMON_NOUNS = [
    "cat", "dog", "bird", "fish", "tree", "book", "car", "house", "phone", "ball",
    "apple", "banana", "orange", "chair", "table", "door", "window", "road", "river",
    "mountain", "forest", "ocean", "desert", "city", "village", "school", "hospital",
    "park", "garden", "beach", "island", "bridge", "tower", "castle", "temple",
]

COMMON_ADJECTIVES = [
    "red", "blue", "green", "yellow", "big", "small", "tall", "short", "old", "new",
    "fast", "slow", "hot", "cold", "soft", "hard", "light", "dark", "loud", "quiet",
    "happy", "sad", "angry", "calm", "bright", "dull", "smooth", "rough", "wet", "dry",
]

COMMON_VERBS = [
    "run", "walk", "jump", "swim", "fly", "eat", "drink", "sleep", "read", "write",
    "sing", "dance", "play", "work", "study", "teach", "learn", "think", "speak", "listen",
]

PERSON_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack",
    "Kate", "Leo", "Maya", "Nick", "Olivia", "Peter", "Quinn", "Rose", "Sam", "Tina",
]

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white"]

COUNTRIES = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Germany", "Berlin"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("UK", "London"), ("China", "Beijing"), ("India", "Delhi"),
    ("Brazil", "Brasilia"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
]

FRUITS = ["apple", "banana", "orange", "grape", "mango", "peach", "pear", "plum", "cherry", "melon"]

ANIMALS = ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "pig", "goat", "duck"]


def get_random_items(rng: random.Random, bank: list, count: int) -> list:
    """Get random items from a bank without replacement."""
    return rng.sample(bank, min(count, len(bank)))


def random_position(rng: random.Random) -> TargetPosition:
    """Return a random target position (START, MIDDLE, or END)."""
    return rng.choice(list(TargetPosition))


def random_words(rng: random.Random, count: int, length: int = 5) -> list[str]:
    """Generate a list of random lowercase words."""
    return [random_word(rng, length) for _ in range(count)]
