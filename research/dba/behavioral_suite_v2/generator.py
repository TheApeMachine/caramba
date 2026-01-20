"""
Test suite generator with template instantiation and stratified sampling.

Generates test cases from templates with:
- Configurable counts per category
- Stratified difficulty sampling
- Position randomization
- Reproducible seeding
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .templates.base import TestCase, TestTemplate, Difficulty, TargetPosition
from .templates import ALL_TEMPLATES


@dataclass
class GenerationConfig:
    """Configuration for test suite generation."""
    seed: int = 42
    default_tests_per_category: int = 30
    stratified_sampling: bool = True
    difficulty_weights: dict[Difficulty, float] = field(default_factory=lambda: {
        Difficulty.EASY: 0.4,
        Difficulty.MEDIUM: 0.35,
        Difficulty.HARD: 0.25,
    })
    position_weights: dict[TargetPosition, float] = field(default_factory=lambda: {
        TargetPosition.START: 0.33,
        TargetPosition.MIDDLE: 0.34,
        TargetPosition.END: 0.33,
    })


@dataclass
class GeneratedSuite:
    """A generated test suite with metadata."""
    tests: list[TestCase]
    config: GenerationConfig
    category_counts: dict[str, int]
    difficulty_distribution: dict[Difficulty, int]
    position_distribution: dict[TargetPosition, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "total_tests": len(self.tests),
            "config": {
                "seed": self.config.seed,
                "default_tests_per_category": self.config.default_tests_per_category,
            },
            "category_counts": self.category_counts,
            "difficulty_distribution": {d.name: c for d, c in self.difficulty_distribution.items()},
            "position_distribution": {p.name: c for p, c in self.position_distribution.items()},
            "tests": [self._test_to_dict(t) for t in self.tests],
        }

    def _test_to_dict(self, test: TestCase) -> dict[str, Any]:
        return {
            "id": test.id,
            "category": test.category,
            "subcategory": test.subcategory,
            "difficulty": test.difficulty.name,
            "prompt": test.prompt,
            "expected": test.expected,
            "kind": test.kind.value,
            "match": test.match,
            "choices": test.choices,
            "target_position": test.target_position.name if test.target_position else None,
            "template_id": test.template_id,
            "seed": test.seed,
        }


class SuiteGenerator:
    """
    Generates behavioral test suites from templates.

    Supports:
    - Stratified sampling by difficulty
    - Position distribution for copy-style tests
    - Per-category count configuration
    - Reproducible generation via seeding
    """

    def __init__(self, config: GenerationConfig | None = None):
        self.config = config or GenerationConfig()
        self.rng = random.Random(self.config.seed)
        self.templates: dict[str, list[TestTemplate]] = defaultdict(list)

    def register_templates(self, templates: list[TestTemplate]) -> None:
        """Register templates organized by category."""
        for template in templates:
            self.templates[template.category].append(template)

    def register_templates_by_category(self, category_templates: dict[str, list[TestTemplate]]) -> None:
        """Register templates from a category -> templates dict."""
        for category, templates in category_templates.items():
            for template in templates:
                self.templates[category].append(template)

    def generate(
        self,
        category_counts: dict[str, int] | None = None,
    ) -> GeneratedSuite:
        """
        Generate a test suite with configured counts per category.

        Args:
            category_counts: Override counts per category. If None, uses default.

        Returns:
            GeneratedSuite with generated tests and metadata.
        """
        all_tests: list[TestCase] = []
        actual_counts: dict[str, int] = {}
        difficulty_dist: dict[Difficulty, int] = defaultdict(int)
        position_dist: dict[TargetPosition, int] = defaultdict(int)

        for category, templates in self.templates.items():
            target_count = (
                category_counts.get(category, self.config.default_tests_per_category)
                if category_counts
                else self.config.default_tests_per_category
            )

            category_tests = self._generate_category(
                category, templates, target_count
            )

            all_tests.extend(category_tests)
            actual_counts[category] = len(category_tests)

            # Track distributions
            for test in category_tests:
                difficulty_dist[test.difficulty] += 1
                if test.target_position:
                    position_dist[test.target_position] += 1

        return GeneratedSuite(
            tests=all_tests,
            config=self.config,
            category_counts=actual_counts,
            difficulty_distribution=dict(difficulty_dist),
            position_distribution=dict(position_dist),
        )

    def _generate_category(
        self,
        category: str,
        templates: list[TestTemplate],
        target_count: int,
    ) -> list[TestCase]:
        """Generate tests for a single category."""
        tests: list[TestCase] = []

        if not templates:
            return tests

        # Generate tests by cycling through templates
        tests_per_template = max(1, target_count // len(templates))
        remainder = target_count % len(templates)

        for i, template in enumerate(templates):
            n = tests_per_template + (1 if i < remainder else 0)
            for _ in range(n):
                seed = self.rng.randint(0, 1000000)
                template_rng = random.Random(seed)
                test = template.generate(template_rng)
                test.prompt = test.prompt.strip() + "\n\nAnswer:"
                test.seed = seed
                test.template_id = f"{category}_{template.__class__.__name__}"
                tests.append(test)

        return tests[:target_count]


def load_all_templates() -> dict[str, list[TestTemplate]]:
    """Load all templates from template modules."""
    return cast(dict[str, list[TestTemplate]], ALL_TEMPLATES)


def generate_suite(
    seed: int = 42,
    tests_per_category: int = 30,
    category_counts: dict[str, int] | None = None,
) -> GeneratedSuite:
    """
    Convenience function to generate a full test suite.

    Args:
        seed: Random seed for reproducibility
        tests_per_category: Default number of tests per category
        category_counts: Optional per-category count overrides

    Returns:
        GeneratedSuite with all tests
    """
    config = GenerationConfig(
        seed=seed,
        default_tests_per_category=tests_per_category,
    )

    generator = SuiteGenerator(config)
    generator.register_templates_by_category(load_all_templates())

    return generator.generate(category_counts)


if __name__ == "__main__":
    # Demo generation
    suite = generate_suite(seed=42, tests_per_category=10)

    print(f"Generated {len(suite.tests)} tests")
    print(f"\nCategory counts:")
    for cat, count in sorted(suite.category_counts.items()):
        print(f"  {cat}: {count}")

    print(f"\nDifficulty distribution:")
    for diff, count in suite.difficulty_distribution.items():
        print(f"  {diff.name}: {count}")

    print(f"\nPosition distribution:")
    for pos, count in suite.position_distribution.items():
        print(f"  {pos.name}: {count}")

    print(f"\nSample tests:")
    for test in suite.tests[:5]:
        print(f"\n--- {test.id} ({test.category}/{test.subcategory}) ---")
        print(f"Prompt:\n{test.prompt[:200]}...")
        print(f"Expected: {test.expected}")
