"""
Distractor test templates - tests attention focus under interference.

Key variations:
- Target position in sequence (start/middle/end)
- Distractor types (explicit, implicit, typo-like, semantic)
- Number of distractors
- Similarity to target
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
    random_upper_word,
    random_number,
    random_alphanumeric,
    COMMON_NOUNS,
    COLORS,
    ANIMALS,
    FRUITS,
    PERSON_NAMES,
)


@dataclass
class ExplicitDistractorTemplate(TestTemplate):
    """
    Explicit distractors with labeled target vs ignore items.

    Format: "Target: X. Ignore: Y, Z, W. What is the target?"
    """

    category = "distractor_tests"
    subcategory = "explicit"

    def __init__(
        self,
        difficulty: Difficulty,
        target_position: TargetPosition,
        distractor_count: int = 3,
    ):
        self.difficulty = difficulty
        self.target_position = target_position
        self.distractor_count = distractor_count
        self.template_id = f"explicit_{difficulty.name.lower()}_{target_position.name.lower()}"

    def generate(self, rng: random.Random) -> TestCase:
        # Generate target and distractors
        content_type = rng.choice(["numbers", "words", "codes"])

        if content_type == "numbers":
            # Generate similar-looking numbers
            base = random_number(rng, 1000, 9999)
            target = str(base)
            distractors = [
                str(base + rng.randint(-100, 100))
                for _ in range(self.distractor_count)
            ]
            # Ensure distractors are different from target
            distractors = [d for d in distractors if d != target][:self.distractor_count]

        elif content_type == "words":
            target = random_upper_word(rng, 4)
            distractors = [random_upper_word(rng, 4) for _ in range(self.distractor_count)]

        else:  # codes
            target = random_alphanumeric(rng, 5)
            distractors = [random_alphanumeric(rng, 5) for _ in range(self.distractor_count)]

        # Build prompt based on target position
        all_items = distractors + [target]
        rng.shuffle(all_items)

        # Reposition target
        all_items.remove(target)
        if self.target_position == TargetPosition.START:
            ordered = [target] + all_items
        elif self.target_position == TargetPosition.END:
            ordered = all_items + [target]
        else:
            mid = len(all_items) // 2
            ordered = all_items[:mid] + [target] + all_items[mid:]

        # Create prompt with various formats
        format_type = rng.choice(["remember_forget", "target_ignore", "correct_wrong"])

        if format_type == "remember_forget":
            prompt = f"""Remember: {target}
Forget: {', '.join(distractors)}
The word to remember is"""
            choices = [f" {target}"] + [f" {d}" for d in distractors]

        elif format_type == "target_ignore":
            prompt = f"""Target: {target}
Ignore: {', '.join(distractors)}
The target is"""
            choices = [f" {target}"] + [f" {d}" for d in distractors]

        else:  # correct_wrong
            prompt = f"""Correct: {target}
Wrong: {', '.join(distractors)}
The correct answer is"""
            choices = [f" {target}"] + [f" {d}" for d in distractors]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {target}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=self.target_position,
        )


@dataclass
class ImplicitDistractorTemplate(TestTemplate):
    """
    Implicit distractors from earlier few-shot examples.

    The model must attend to the LAST example, not earlier ones.
    """

    category = "distractor_tests"
    subcategory = "implicit"

    def __init__(
        self,
        difficulty: Difficulty,
        target_position: TargetPosition,
        num_examples: int = 4,
    ):
        self.difficulty = difficulty
        self.target_position = target_position
        self.num_examples = num_examples
        self.template_id = f"implicit_{difficulty.name.lower()}_{target_position.name.lower()}"

    def generate(self, rng: random.Random) -> TestCase:
        # Generate items for few-shot
        content_type = rng.choice(["colors", "numbers", "words"])

        if content_type == "colors":
            items = rng.sample(COLORS, self.num_examples)
        elif content_type == "numbers":
            items = [str(random_number(rng, 100, 999)) for _ in range(self.num_examples)]
        else:
            items = [random_upper_word(rng, 4) for _ in range(self.num_examples)]

        # Determine target based on position
        if self.target_position == TargetPosition.START:
            # Target is first, but query asks for it
            target_idx = 0
        elif self.target_position == TargetPosition.END:
            target_idx = -1
        else:  # MIDDLE
            target_idx = len(items) // 2

        target = items[target_idx]

        # Build few-shot prompt - the last item should be copied
        lines = []
        for item in items[:-1]:
            lines.append(f"Item: {item} -> {item}")
        lines.append(f"Item: {items[-1]} ->")

        prompt = "\n".join(lines)

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=items[-1],  # Should copy the LAST item
            kind=EvalKind.EXACT_MATCH_GREEDY,
            target_position=self.target_position,
        )


@dataclass
class TypoDistractorTemplate(TestTemplate):
    """
    Typo-like distractors that are very similar to target.

    Tests fine-grained discrimination.
    """

    category = "distractor_tests"
    subcategory = "typo"

    def __init__(self, difficulty: Difficulty):
        self.difficulty = difficulty
        self.template_id = f"typo_{difficulty.name.lower()}"

    def _generate_typos(self, rng: random.Random, original: str, count: int) -> list[str]:
        """Generate typo variants of a string."""
        typos = []
        for _ in range(count * 2):  # Generate extra, dedupe later
            variant = list(original)
            operation = rng.choice(["swap", "replace", "delete", "insert"])

            if operation == "swap" and len(variant) > 1:
                i = rng.randint(0, len(variant) - 2)
                variant[i], variant[i + 1] = variant[i + 1], variant[i]
            elif operation == "replace":
                i = rng.randint(0, len(variant) - 1)
                if variant[i].isdigit():
                    variant[i] = str(rng.randint(0, 9))
                else:
                    variant[i] = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            elif operation == "delete" and len(variant) > 2:
                i = rng.randint(0, len(variant) - 1)
                del variant[i]
            else:  # insert
                i = rng.randint(0, len(variant))
                char = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                variant.insert(i, char)

            result = ''.join(variant)
            if result != original and result not in typos:
                typos.append(result)

        return typos[:count]

    def generate(self, rng: random.Random) -> TestCase:
        # Generate target
        content_type = rng.choice(["number", "code"])

        if content_type == "number":
            target = str(random_number(rng, 10000, 99999))
        else:
            target = random_alphanumeric(rng, 6)

        # Generate typo distractors
        distractors = self._generate_typos(rng, target, 3)

        # Ensure we have enough distractors
        while len(distractors) < 3:
            if content_type == "number":
                distractors.append(str(random_number(rng, 10000, 99999)))
            else:
                distractors.append(random_alphanumeric(rng, 6))

        format_type = rng.choice(["code", "word", "find"])

        if format_type == "code":
            prompt = f"""Code: {target}
Similar codes: {', '.join(distractors)}
The correct code is"""
        elif format_type == "word":
            prompt = f"""Word: {target}
Misspellings: {', '.join(distractors)}
The correct spelling is"""
        else:
            prompt = f"""Find the exact match: {target}
Options: {target}, {', '.join(distractors)}
Exact match:"""

        choices = [f" {target}"] + [f" {d}" for d in distractors[:3]]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {target}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
        )


@dataclass
class SemanticDistractorTemplate(TestTemplate):
    """
    Semantically related distractors.

    Target is from same category as distractors (e.g., all animals).
    """

    category = "distractor_tests"
    subcategory = "semantic"

    def __init__(self, difficulty: Difficulty, target_position: TargetPosition):
        self.difficulty = difficulty
        self.target_position = target_position
        self.template_id = f"semantic_{difficulty.name.lower()}_{target_position.name.lower()}"

    def generate(self, rng: random.Random) -> TestCase:
        # Choose a semantic category
        category = rng.choice(["animals", "colors", "fruits", "names"])

        if category == "animals":
            pool = ANIMALS
            category_name = "animal"
        elif category == "colors":
            pool = COLORS
            category_name = "color"
        elif category == "fruits":
            pool = FRUITS
            category_name = "fruit"
        else:
            pool = PERSON_NAMES
            category_name = "name"

        # Select target and distractors
        selected = rng.sample(pool, 4)
        target = selected[0]
        distractors = selected[1:]

        # Arrange based on position
        if self.target_position == TargetPosition.START:
            mention_order = [target] + distractors
        elif self.target_position == TargetPosition.END:
            mention_order = distractors + [target]
        else:
            mention_order = [distractors[0], target, distractors[1], distractors[2]]

        prompt = f"""The {category_name} is: {target.upper()}
Other {category_name}s: {', '.join(d.upper() for d in distractors)}
What is the {category_name}?"""

        choices = [f" {target.upper()}"] + [f" {d.upper()}" for d in distractors]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {target.upper()}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=self.target_position,
        )


@dataclass
class PositionalDistractorTemplate(TestTemplate):
    """
    Test retrieval with target at various positions in a sequence.

    Critical for testing if model has positional bias.
    """

    category = "distractor_tests"
    subcategory = "positional"

    def __init__(self, difficulty: Difficulty, target_position: TargetPosition):
        self.difficulty = difficulty
        self.target_position = target_position
        self.template_id = f"positional_{difficulty.name.lower()}_{target_position.name.lower()}"

        self.sequence_lengths = {
            Difficulty.EASY: 5,
            Difficulty.MEDIUM: 8,
            Difficulty.HARD: 12,
        }

    def generate(self, rng: random.Random) -> TestCase:
        seq_len = self.sequence_lengths[self.difficulty]

        # Generate unique items
        items = [random_upper_word(rng, 3) for _ in range(seq_len)]

        # Pick target position index
        if self.target_position == TargetPosition.START:
            target_idx = 0
        elif self.target_position == TargetPosition.END:
            target_idx = seq_len - 1
        else:  # MIDDLE
            target_idx = seq_len // 2

        target = items[target_idx]

        # Create prompt asking for item at specific position
        position_words = {0: "first", seq_len - 1: "last"}
        if target_idx in position_words:
            position_desc = position_words[target_idx]
        else:
            position_desc = f"{target_idx + 1}th"

        prompt = f"""Sequence: {', '.join(items)}
The {position_desc} item is"""

        choices = [f" {target}"] + [f" {items[i]}" for i in range(len(items)) if i != target_idx][:3]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {target}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=self.target_position,
        )


@dataclass
class NoisyRetrievalTemplate(TestTemplate):
    """
    Target buried in irrelevant noise text.

    Tests attention's ability to focus on relevant information.
    """

    category = "distractor_tests"
    subcategory = "noisy_retrieval"

    def __init__(self, difficulty: Difficulty, target_position: TargetPosition):
        self.difficulty = difficulty
        self.target_position = target_position
        self.template_id = f"noisy_{difficulty.name.lower()}_{target_position.name.lower()}"

        self.noise_amounts = {
            Difficulty.EASY: 2,
            Difficulty.MEDIUM: 5,
            Difficulty.HARD: 10,
        }

    def generate(self, rng: random.Random) -> TestCase:
        target = random_upper_word(rng, 5)
        noise_lines = self.noise_amounts[self.difficulty]

        # Generate noise
        noise_templates = [
            "NOISE NOISE NOISE",
            "xxx yyy zzz",
            "---",
            "...",
            f"{random_word(rng, 3)} {random_word(rng, 3)} {random_word(rng, 3)}",
            "IRRELEVANT TEXT",
            "IGNORE THIS LINE",
        ]

        noise = [rng.choice(noise_templates) for _ in range(noise_lines)]

        # Place target
        target_line = f"TARGET: {target}"

        if self.target_position == TargetPosition.START:
            lines = [target_line] + noise
        elif self.target_position == TargetPosition.END:
            lines = noise + [target_line]
        else:
            mid = len(noise) // 2
            lines = noise[:mid] + [target_line] + noise[mid:]

        lines.append("TARGET is")
        prompt = "\n".join(lines)

        choices = [f" {target}", f" NOISE", f" xxx", f" IRRELEVANT"]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {target}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=self.target_position,
        )


def get_all_distractor_templates() -> list[TestTemplate]:
    """Get all distractor test templates."""
    templates = []

    for difficulty in Difficulty:
        for position in TargetPosition:
            templates.append(ExplicitDistractorTemplate(difficulty, position))
            templates.append(ImplicitDistractorTemplate(difficulty, position))
            templates.append(SemanticDistractorTemplate(difficulty, position))
            templates.append(PositionalDistractorTemplate(difficulty, position))
            templates.append(NoisyRetrievalTemplate(difficulty, position))

        # Typo doesn't need position variation
        templates.append(TypoDistractorTemplate(difficulty))

    return templates


# Export TEMPLATES for consistency with other template modules
TEMPLATES = get_all_distractor_templates()
