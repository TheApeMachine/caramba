"""
Few-shot learning templates.

Tests in-context learning ability by providing examples and asking
the model to apply the pattern to new inputs.
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
    random_position,
    random_word,
    random_words,
)


class SimpleTransformTemplate(TestTemplate):
    """Few-shot pattern: apply a simple string transformation."""

    category = "fewshot_learning"
    subcategory = "simple_transform"

    TRANSFORMS = [
        ("uppercase", lambda s: s.upper()),
        ("lowercase", lambda s: s.lower()),
        ("reverse", lambda s: s[::-1]),
        ("first_letter", lambda s: s[0] if s else ""),
        ("last_letter", lambda s: s[-1] if s else ""),
        ("double", lambda s: s + s),
        ("first_three", lambda s: s[:3]),
    ]

    def generate(self, rng: random.Random) -> TestCase:
        transform_name, transform_fn = rng.choice(self.TRANSFORMS)
        difficulty = rng.choice(list(Difficulty))

        # Generate examples
        if difficulty == Difficulty.EASY:
            num_examples = 3
            words = ["cat", "dog", "sun", "moon", "tree", "bird"]
        elif difficulty == Difficulty.MEDIUM:
            num_examples = 2
            words = ["apple", "banana", "orange", "grape", "melon"]
        else:
            num_examples = 2
            words = ["elephant", "giraffe", "penguin", "dolphin"]

        rng.shuffle(words)
        example_words = words[:num_examples]
        test_word = words[num_examples] if len(words) > num_examples else rng.choice(words)

        # Build prompt
        examples = "\n".join([f"{w} -> {transform_fn(w)}" for w in example_words])
        prompt = f"{examples}\n{test_word} ->"
        expected = transform_fn(test_word)

        return TestCase(
            id=f"fewshot_{transform_name}_{test_word}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={
                "transform": transform_name,
                "test_word": test_word,
                # Explicitly enable "presence-contained" soft credit in weighted scoring
                # for few-shot learning tasks.
                "allow_presence_contained": True,
            },
        )


class PrefixSuffixTemplate(TestTemplate):
    """Few-shot pattern: add prefix or suffix to words."""

    category = "fewshot_learning"
    subcategory = "prefix_suffix"

    PATTERNS = [
        ("un_prefix", "un", True),
        ("re_prefix", "re", True),
        ("pre_prefix", "pre", True),
        ("ing_suffix", "ing", False),
        ("ed_suffix", "ed", False),
        ("ly_suffix", "ly", False),
        ("ness_suffix", "ness", False),
    ]

    def generate(self, rng: random.Random) -> TestCase:
        pattern_name, affix, is_prefix = rng.choice(self.PATTERNS)
        difficulty = rng.choice(list(Difficulty))

        if is_prefix:
            transform = lambda w: affix + w
        else:
            transform = lambda w: w + affix

        if difficulty == Difficulty.EASY:
            num_examples = 3
            words = ["happy", "kind", "fair", "clear", "safe"]
        elif difficulty == Difficulty.MEDIUM:
            num_examples = 2
            words = ["certain", "direct", "complete", "perfect"]
        else:
            num_examples = 2
            words = ["fortunate", "probable", "reasonable", "comfortable"]

        rng.shuffle(words)
        example_words = words[:num_examples]
        test_word = words[num_examples] if len(words) > num_examples else words[-1]

        examples = "\n".join([f"{w} -> {transform(w)}" for w in example_words])
        prompt = f"{examples}\n{test_word} ->"
        expected = transform(test_word)

        return TestCase(
            id=f"fewshot_{pattern_name}_{test_word}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={
                "pattern": pattern_name,
                "test_word": test_word,
                # Explicitly enable "presence-contained" soft credit in weighted scoring
                # for few-shot learning tasks.
                "allow_presence_contained": True,
            },
        )


class ArithmeticPatternTemplate(TestTemplate):
    """Few-shot pattern: arithmetic operations."""

    category = "fewshot_learning"
    subcategory = "arithmetic_pattern"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Simple addition
            op = "add"
            delta = rng.randint(1, 5)
            examples = [(i, i + delta) for i in rng.sample(range(1, 20), 3)]
            test_input = rng.randint(1, 20)
            expected = str(test_input + delta)
        elif difficulty == Difficulty.MEDIUM:
            # Multiplication
            op = "multiply"
            factor = rng.randint(2, 5)
            examples = [(i, i * factor) for i in rng.sample(range(1, 15), 2)]
            test_input = rng.randint(1, 15)
            expected = str(test_input * factor)
        else:
            # Square or more complex
            op = "square"
            examples = [(i, i * i) for i in rng.sample(range(2, 10), 2)]
            test_input = rng.randint(2, 10)
            expected = str(test_input * test_input)

        examples_str = "\n".join([f"{a} -> {b}" for a, b in examples])
        prompt = f"{examples_str}\n{test_input} ->"

        return TestCase(
            id=f"fewshot_{op}_{test_input}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={
                "operation": op,
                "test_input": test_input,
                # Explicitly enable "presence-contained" soft credit in weighted scoring
                # for few-shot learning tasks.
                "allow_presence_contained": True,
            },
        )


class SymbolMappingTemplate(TestTemplate):
    """Few-shot pattern: map symbols to other symbols."""

    category = "fewshot_learning"
    subcategory = "symbol_mapping"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Letter to number
            mapping = {chr(ord('a') + i): str(i + 1) for i in range(10)}
            num_examples = 3
        elif difficulty == Difficulty.MEDIUM:
            # Custom symbol mapping
            symbols = ['@', '#', '$', '%', '&', '*', '+', '=']
            outputs = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta']
            mapping = dict(zip(symbols, outputs))
            num_examples = 2
        else:
            # Emoji-like or special chars
            symbols = ['()', '[]', '{}', '<>', '||', '//', '\\\\', '**']
            outputs = ['paren', 'bracket', 'brace', 'angle', 'pipe', 'slash', 'back', 'star']
            mapping = dict(zip(symbols, outputs))
            num_examples = 2

        items = list(mapping.items())
        rng.shuffle(items)
        example_items = items[:num_examples]
        test_item = items[num_examples] if len(items) > num_examples else items[-1]

        examples = "\n".join([f"{k} -> {v}" for k, v in example_items])
        prompt = f"{examples}\n{test_item[0]} ->"
        expected = test_item[1]

        return TestCase(
            id=f"fewshot_symbol_{test_item[0][:2]}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={
                "test_symbol": test_item[0],
                # Explicitly enable "presence-contained" soft credit in weighted scoring
                # for few-shot learning tasks.
                "allow_presence_contained": True,
            },
        )


class CategoryClassificationTemplate(TestTemplate):
    """Few-shot pattern: classify items into categories."""

    category = "fewshot_learning"
    subcategory = "classification"

    CATEGORIES = {
        "fruit_vegetable": {
            "fruit": ["apple", "banana", "orange", "grape", "mango"],
            "vegetable": ["carrot", "broccoli", "spinach", "potato", "onion"],
        },
        "animal_plant": {
            "animal": ["dog", "cat", "elephant", "bird", "fish"],
            "plant": ["rose", "oak", "tulip", "fern", "cactus"],
        },
        "hot_cold": {
            "hot": ["fire", "sun", "lava", "oven", "desert"],
            "cold": ["ice", "snow", "arctic", "freezer", "glacier"],
        },
    }

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        category_type = rng.choice(list(self.CATEGORIES.keys()))
        categories = self.CATEGORIES[category_type]

        cat_names = list(categories.keys())
        all_items = []
        for cat, items in categories.items():
            for item in items:
                all_items.append((item, cat))

        rng.shuffle(all_items)

        if difficulty == Difficulty.EASY:
            num_examples = 4
        elif difficulty == Difficulty.MEDIUM:
            num_examples = 3
        else:
            num_examples = 2

        example_items = all_items[:num_examples]
        test_item = all_items[num_examples]

        examples = "\n".join([f"{item} -> {cat}" for item, cat in example_items])
        prompt = f"{examples}\n{test_item[0]} ->"
        expected = test_item[1]

        return TestCase(
            id=f"fewshot_classify_{test_item[0]}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={
                "category_type": category_type,
                "test_item": test_item[0],
                # Explicitly enable "presence-contained" soft credit in weighted scoring
                # for few-shot learning tasks.
                "allow_presence_contained": True,
            },
        )


class WordRelationTemplate(TestTemplate):
    """Few-shot pattern: word relationships (analogies)."""

    category = "fewshot_learning"
    subcategory = "word_relation"

    RELATIONS = [
        # (a, b) pairs with same relationship
        [("dog", "puppy"), ("cat", "kitten"), ("cow", "calf"), ("horse", "foal")],
        [("big", "small"), ("hot", "cold"), ("up", "down"), ("fast", "slow")],
        [("king", "queen"), ("man", "woman"), ("boy", "girl"), ("father", "mother")],
        [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo")],
    ]

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        relation_set = rng.choice(self.RELATIONS)
        rng.shuffle(relation_set)

        if difficulty == Difficulty.EASY:
            num_examples = 3
        elif difficulty == Difficulty.MEDIUM:
            num_examples = 2
        else:
            num_examples = 2

        example_pairs = relation_set[:num_examples]
        test_pair = relation_set[num_examples]

        examples = "\n".join([f"{a} : {b}" for a, b in example_pairs])
        prompt = f"{examples}\n{test_pair[0]} :"
        expected = test_pair[1]

        return TestCase(
            id=f"fewshot_relation_{test_pair[0]}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={
                "test_word": test_pair[0],
                # Explicitly enable "presence-contained" soft credit in weighted scoring
                # for few-shot learning tasks.
                "allow_presence_contained": True,
            },
        )


# Export all templates
TEMPLATES = [
    SimpleTransformTemplate(),
    PrefixSuffixTemplate(),
    ArithmeticPatternTemplate(),
    SymbolMappingTemplate(),
    CategoryClassificationTemplate(),
    WordRelationTemplate(),
]
