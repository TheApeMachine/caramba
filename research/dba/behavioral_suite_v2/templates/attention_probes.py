"""
Attention probe templates.

Tests designed to probe attention patterns - requiring fine-grained
discrimination, focus on specific tokens, and handling of similar items.
These tests are particularly useful for analyzing attention mechanisms.
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
)


class FineDiscriminationTemplate(TestTemplate):
    """Discriminate between very similar options."""

    category = "attention_probes"
    subcategory = "fine_discrimination"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Numbers differing by 1
            base = rng.randint(100, 900)
            options = [base, base + 1, base - 1]
            target = base
            rng.shuffle(options)
            prompt = f"Which of these equals {target}: {', '.join(map(str, options))}?"
            expected = str(target)

        elif difficulty == Difficulty.MEDIUM:
            # Words differing by one letter
            pairs = [
                ("cat", "car", "cap"),
                ("dog", "dot", "log"),
                ("bat", "bag", "bad"),
                ("pen", "pet", "peg"),
            ]
            words = rng.choice(pairs)
            target = words[0]
            prompt = f"The word '{target}' is which of these: {', '.join(words)}?"
            expected = target

        else:
            # Numbers with similar digits
            base = rng.randint(10, 50)
            n1 = base * 11  # e.g., 33
            n2 = base * 11 + 1  # e.g., 34
            n3 = base * 11 - 1  # e.g., 32
            target = n1
            options = [n1, n2, n3]
            rng.shuffle(options)
            prompt = f"Find {target} in this list: {', '.join(map(str, options))}"
            expected = str(target)

        return TestCase(
            id=f"attn_fine_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "fine_discrimination"},
        )


class NoisyRetrievalTemplate(TestTemplate):
    """Retrieve target from among similar distractors."""

    category = "attention_probes"
    subcategory = "noisy_retrieval"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        position = random_position(rng)

        if difficulty == Difficulty.EASY:
            # Target number among different numbers
            target = rng.randint(10, 99)
            distractors = [rng.randint(10, 99) for _ in range(3)]
            while target in distractors:
                distractors = [rng.randint(10, 99) for _ in range(3)]

        elif difficulty == Difficulty.MEDIUM:
            # Target among similar numbers
            target = rng.randint(100, 500)
            distractors = [target + rng.choice([-10, -5, 5, 10, 15, -15]) for _ in range(4)]

        else:
            # Target among very similar numbers
            target = rng.randint(1000, 5000)
            distractors = [target + rng.choice([-1, 1, -2, 2, -3, 3]) for _ in range(5)]

        # Place target at specified position
        all_items = list(distractors)
        if position == TargetPosition.START:
            all_items.insert(0, target)
        elif position == TargetPosition.END:
            all_items.append(target)
        else:
            mid = len(all_items) // 2
            all_items.insert(mid, target)

        items_str = ", ".join(map(str, all_items))
        prompt = f"Find {target} in this list and confirm it exists: {items_str}"
        expected = str(target)

        return TestCase(
            id=f"attn_noisy_{target}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=position,
            metadata={"target": target, "num_distractors": len(distractors)},
        )


class RecentVsDistantTemplate(TestTemplate):
    """Test attention to recent vs distant context."""

    category = "attention_probes"
    subcategory = "recent_vs_distant"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        values = [rng.randint(10, 99) for _ in range(5)]

        if difficulty == Difficulty.EASY:
            # Ask about most recent
            facts = [f"{names[i]}'s number is {values[i]}." for i in range(3)]
            prompt = " ".join(facts) + f" What is {names[2]}'s number?"
            expected = str(values[2])
            target_pos = TargetPosition.END

        elif difficulty == Difficulty.MEDIUM:
            # Ask about first (distant)
            facts = [f"{names[i]}'s number is {values[i]}." for i in range(4)]
            prompt = " ".join(facts) + f" What is {names[0]}'s number?"
            expected = str(values[0])
            target_pos = TargetPosition.START

        else:
            # Ask about middle with more context
            facts = [f"{names[i]}'s number is {values[i]}." for i in range(5)]
            prompt = " ".join(facts) + f" What is {names[2]}'s number?"
            expected = str(values[2])
            target_pos = TargetPosition.MIDDLE

        return TestCase(
            id=f"attn_distance_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=target_pos,
            metadata={"type": "recent_vs_distant"},
        )


class SimilarItemsTemplate(TestTemplate):
    """Distinguish between items with similar attributes."""

    category = "attention_probes"
    subcategory = "similar_items"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Two items, one attribute
            items = [
                ("red apple", "blue apple"),
                ("small dog", "large dog"),
                ("old book", "new book"),
            ]
            pair = rng.choice(items)
            attr = pair[0].split()[0]  # "red", "small", "old"
            prompt = f"I have a {pair[0]} and a {pair[1]}. Which one is {attr}?"
            expected = pair[0]

        elif difficulty == Difficulty.MEDIUM:
            # Three items
            colors = ["red", "blue", "green"]
            objects = ["ball", "box", "cup"]
            assignments = list(zip(colors, objects))
            rng.shuffle(assignments)
            descriptions = [f"a {c} {o}" for c, o in assignments]
            target_color, target_obj = assignments[0]
            prompt = f"I have {', '.join(descriptions)}. What color is the {target_obj}?"
            expected = target_color

        else:
            # More items with similar names
            people = [
                ("John Smith", "engineer"),
                ("John Brown", "teacher"),
                ("Jane Smith", "doctor"),
            ]
            rng.shuffle(people)
            facts = [f"{name} is a {job}." for name, job in people]
            target = people[1]
            prompt = " ".join(facts) + f" What is {target[0]}'s job?"
            expected = target[1]

        return TestCase(
            id=f"attn_similar_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=random_position(rng),
            metadata={"type": "similar_items"},
        )


class BindingProbeTemplate(TestTemplate):
    """Probe entity-attribute binding in attention."""

    category = "attention_probes"
    subcategory = "binding_probe"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Simple binding
            entities = [("cat", "black"), ("dog", "white")]
            rng.shuffle(entities)
            facts = [f"The {e} is {c}." for e, c in entities]
            target = entities[0]
            prompt = " ".join(facts) + f" What color is the {target[0]}?"
            expected = target[1]

        elif difficulty == Difficulty.MEDIUM:
            # More entities
            entities = [
                ("Alice", "Paris"),
                ("Bob", "London"),
                ("Carol", "Tokyo"),
            ]
            rng.shuffle(entities)
            facts = [f"{n} lives in {c}." for n, c in entities]
            target = entities[1]
            prompt = " ".join(facts) + f" Where does {target[0]} live?"
            expected = target[1]

        else:
            # Crossed bindings (same attribute value for multiple entities)
            entities = [
                ("Alice", "25", "Paris"),
                ("Bob", "25", "London"),
                ("Carol", "30", "Paris"),
            ]
            rng.shuffle(entities)
            facts = [f"{n} is {a} years old and lives in {c}." for n, a, c in entities]
            # Ask about the one with unique combination
            target = entities[0]
            prompt = " ".join(facts) + f" How old is {target[0]}?"
            expected = target[1]

        return TestCase(
            id=f"attn_binding_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=random_position(rng),
            metadata={"type": "binding_probe"},
        )


class OrderingProbeTemplate(TestTemplate):
    """Probe attention to sequential order."""

    category = "attention_probes"
    subcategory = "ordering_probe"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # First/last of short sequence
            items = rng.sample(["apple", "banana", "cherry", "date"], 3)
            if rng.random() < 0.5:
                prompt = f"What is the first item: {', '.join(items)}?"
                expected = items[0]
            else:
                prompt = f"What is the last item: {', '.join(items)}?"
                expected = items[-1]

        elif difficulty == Difficulty.MEDIUM:
            # Nth item
            items = rng.sample(["red", "blue", "green", "yellow", "purple"], 4)
            n = rng.randint(2, len(items) - 1)
            prompt = f"What is item #{n}: {', '.join(items)}?"
            expected = items[n - 1]

        else:
            # Order with distractors
            items = [str(rng.randint(10, 99)) for _ in range(5)]
            n = rng.randint(2, 4)
            filler = "Remember to check carefully."
            prompt = f"Given the list: {', '.join(items)}. {filler} What is item #{n}?"
            expected = items[n - 1]

        return TestCase(
            id=f"attn_order_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "ordering_probe"},
        )


class InterferenceProbeTemplate(TestTemplate):
    """Test attention under interference from similar content."""

    category = "attention_probes"
    subcategory = "interference"

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Repeat with variation
            base = rng.randint(10, 50)
            target = base * 2
            interferer = base * 2 + 1
            prompt = f"Double {base}. Not {interferer}, but the actual double of {base} is"
            expected = str(target)

        elif difficulty == Difficulty.MEDIUM:
            # Multiple similar facts
            name = rng.choice(["Alice", "Bob", "Charlie"])
            correct_value = rng.randint(20, 40)
            wrong_values = [correct_value + i for i in [-2, -1, 1, 2]]
            wrong_facts = [f"Some say {name} is {v}." for v in wrong_values[:2]]
            correct_fact = f"Actually, {name} is {correct_value}."
            prompt = " ".join(wrong_facts) + " " + correct_fact + f" What is {name}'s true value?"
            expected = str(correct_value)

        else:
            # Heavy interference
            target = rng.randint(100, 500)
            similar = [target + rng.randint(-5, 5) for _ in range(4)]
            similar = [s for s in similar if s != target][:3]
            mentions = [f"Not {s}." for s in similar]
            prompt = f"The answer is {target}. " + " ".join(mentions) + f" What is the answer?"
            expected = str(target)

        return TestCase(
            id=f"attn_interfere_{difficulty.value}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=random_position(rng),
            metadata={"type": "interference"},
        )


# Export all templates
TEMPLATES = [
    FineDiscriminationTemplate(),
    NoisyRetrievalTemplate(),
    RecentVsDistantTemplate(),
    SimilarItemsTemplate(),
    BindingProbeTemplate(),
    OrderingProbeTemplate(),
    InterferenceProbeTemplate(),
]
