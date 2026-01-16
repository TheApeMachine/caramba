"""
Copy task templates - tests raw memorization fidelity.

Key randomization:
- Target position (start/middle/end of few-shot examples)
- Sequence length (easy: 3-5, medium: 6-10, hard: 11-20)
- Content type (letters, numbers, alphanumeric, mixed)
- Format variations (Copy:, Echo:, Repeat:, ->)
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
    random_alphanumeric,
    random_number,
)


@dataclass
class CopySimpleTemplate(TestTemplate):
    """Copy single token/word with few-shot examples."""

    category = "copy_tasks"
    subcategory = "simple"

    def __init__(
        self,
        difficulty: Difficulty,
        target_position: TargetPosition,
        content_type: str = "upper",  # upper, lower, number, alpha
        format_style: str = "copy",   # copy, echo, repeat, arrow
    ):
        self.difficulty = difficulty
        self.target_position = target_position
        self.content_type = content_type
        self.format_style = format_style
        self.template_id = f"copy_simple_{content_type}_{format_style}_{target_position.name.lower()}"

    def _generate_item(self, rng: random.Random) -> str:
        if self.content_type == "upper":
            return random_upper_word(rng, 3)
        elif self.content_type == "lower":
            return random_word(rng, 4)
        elif self.content_type == "number":
            return str(random_number(rng, 100, 999))
        else:  # alpha
            return random_alphanumeric(rng, 4)

    def _format_line(self, item: str, with_answer: bool = True) -> str:
        formats = {
            "copy": f"Copy: {item}" if with_answer else "Copy:",
            "echo": f"Echo: {item}" if with_answer else "Echo:",
            "repeat": f"Repeat: {item}" if with_answer else "Repeat:",
            "arrow": f"{item} ->" if with_answer else "->",
        }
        return formats.get(self.format_style, f"Copy: {item}" if with_answer else "Copy:")

    def generate(self, rng: random.Random) -> TestCase:
        # Generate 3 examples + 1 query
        items = [self._generate_item(rng) for _ in range(4)]
        target = items[-1]

        # Create examples based on target position
        if self.target_position == TargetPosition.START:
            examples = items[:3]
            query_item = items[0]
        elif self.target_position == TargetPosition.END:
            examples = items[:3]
            query_item = items[-1]
        else:  # MIDDLE
            examples = [items[0], items[2], items[1]]  # Swap to put "target pattern" in middle
            query_item = items[-1]

        # Build prompt
        lines = []
        for item in examples:
            lines.append(self._format_line(item))
            lines.append(self._format_line(item))  # Answer line

        # Actually, let's do it differently - standard few-shot format
        lines = []
        for i, item in enumerate(items[:-1]):
            lines.append(f"{self._get_prefix()}{item}")
        lines.append(f"{self._get_prefix()}")

        prompt = "\n".join([
            f"{self._get_prefix()}{items[0]}",
            f"{self._get_prefix()}{items[1]}",
            f"{self._get_prefix()}{items[2]}",
            f"{self._get_prefix()}",
        ])

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=items[2],  # Copy last shown item
            kind=EvalKind.EXACT_MATCH_GREEDY,
            target_position=self.target_position,
        )

    def _get_prefix(self) -> str:
        prefixes = {
            "copy": "Copy: ",
            "echo": "Echo: ",
            "repeat": "Repeat: ",
            "arrow": "",
        }
        return prefixes.get(self.format_style, "Copy: ")


@dataclass
class CopySequenceTemplate(TestTemplate):
    """Copy multi-token sequences."""

    category = "copy_tasks"
    subcategory = "sequence"

    def __init__(
        self,
        difficulty: Difficulty,
        target_position: TargetPosition,
    ):
        self.difficulty = difficulty
        self.target_position = target_position
        self.template_id = f"copy_sequence_{difficulty.name.lower()}_{target_position.name.lower()}"

        # Sequence length based on difficulty
        self.length_ranges = {
            Difficulty.EASY: (3, 5),
            Difficulty.MEDIUM: (6, 10),
            Difficulty.HARD: (11, 15),
        }

    def _generate_sequence(self, rng: random.Random) -> str:
        min_len, max_len = self.length_ranges[self.difficulty]
        length = rng.randint(min_len, max_len)

        # Mix of content types
        content_type = rng.choice(["letters", "numbers", "mixed"])

        if content_type == "letters":
            items = [chr(ord('A') + i) for i in range(length)]
        elif content_type == "numbers":
            items = [str(i + 1) for i in range(length)]
        else:
            items = []
            for i in range(length):
                if i % 2 == 0:
                    items.append(chr(ord('A') + (i // 2)))
                else:
                    items.append(str((i // 2) + 1))

        return " ".join(items)

    def generate(self, rng: random.Random) -> TestCase:
        # Generate sequences
        sequences = [self._generate_sequence(rng) for _ in range(3)]

        # Place target based on position
        if self.target_position == TargetPosition.START:
            target = sequences[0]
            prompt_seqs = sequences
        elif self.target_position == TargetPosition.END:
            target = sequences[-1]
            prompt_seqs = sequences
        else:  # MIDDLE
            target = sequences[1]
            prompt_seqs = [sequences[0], sequences[2], sequences[1]]

        # Build prompt with varying formats
        format_type = rng.choice(["echo", "repeat", "sequence"])
        prefix = {"echo": "Echo:", "repeat": "Repeat:", "sequence": "Sequence:"}[format_type]

        lines = []
        for seq in prompt_seqs[:-1]:
            lines.append(f"{prefix} {seq}")
            lines.append(f"{prefix} {seq}")
        lines.append(f"{prefix} {prompt_seqs[-1]}")
        lines.append(f"{prefix}")

        prompt = "\n".join(lines)

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=prompt_seqs[-1],
            kind=EvalKind.EXACT_MATCH_GREEDY,
            target_position=self.target_position,
        )


@dataclass
class CopyMixedContentTemplate(TestTemplate):
    """Copy with mixed content types and special characters."""

    category = "copy_tasks"
    subcategory = "mixed"

    def __init__(
        self,
        difficulty: Difficulty,
        target_position: TargetPosition,
    ):
        self.difficulty = difficulty
        self.target_position = target_position
        self.template_id = f"copy_mixed_{difficulty.name.lower()}_{target_position.name.lower()}"

    def _generate_mixed(self, rng: random.Random) -> str:
        """Generate mixed content string."""
        templates = [
            lambda: f"{random_upper_word(rng, 2)}-{random_number(rng, 10, 99)}",
            lambda: f"{random_number(rng, 1, 9)}{random_upper_word(rng, 2)}{random_number(rng, 1, 9)}",
            lambda: f"@{random_word(rng, 3)}#{random_number(rng, 10, 99)}",
            lambda: f"[{random_upper_word(rng, 3)}]",
            lambda: f"({random_number(rng, 100, 999)})",
        ]
        return rng.choice(templates)()

    def generate(self, rng: random.Random) -> TestCase:
        items = [self._generate_mixed(rng) for _ in range(4)]

        # Arrange based on target position
        if self.target_position == TargetPosition.START:
            ordered = items
            target = items[0]
        elif self.target_position == TargetPosition.END:
            ordered = items
            target = items[-1]
        else:
            mid_idx = len(items) // 2
            target = items[mid_idx]
            ordered = items[:mid_idx] + items[mid_idx+1:] + [target]

        lines = []
        for item in ordered[:-1]:
            lines.append(f"Data: {item}")
            lines.append(f"Copy: {item}")
        lines.append(f"Data: {ordered[-1]}")
        lines.append("Copy:")

        prompt = "\n".join(lines)

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=ordered[-1],
            kind=EvalKind.EXACT_MATCH_GREEDY,
            target_position=self.target_position,
        )


def get_all_copy_templates() -> list[TestTemplate]:
    """Get all copy task templates with all variations."""
    templates = []

    # Simple copy - all combinations
    for difficulty in Difficulty:
        for position in TargetPosition:
            for content in ["upper", "lower", "number", "alpha"]:
                for style in ["copy", "echo", "repeat"]:
                    templates.append(CopySimpleTemplate(
                        difficulty=difficulty,
                        target_position=position,
                        content_type=content,
                        format_style=style,
                    ))

    # Sequence copy
    for difficulty in Difficulty:
        for position in TargetPosition:
            templates.append(CopySequenceTemplate(
                difficulty=difficulty,
                target_position=position,
            ))

    # Mixed content copy
    for difficulty in Difficulty:
        for position in TargetPosition:
            templates.append(CopyMixedContentTemplate(
                difficulty=difficulty,
                target_position=position,
            ))

    return templates


# Export TEMPLATES for consistency with other template modules
TEMPLATES = get_all_copy_templates()
