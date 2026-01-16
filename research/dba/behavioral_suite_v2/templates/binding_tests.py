"""
Entity-attribute binding test templates.

Tests the model's ability to correctly associate entities with their attributes
under various interference conditions:
- Two-entity binding (A has X, B has Y)
- Three-entity binding
- Attribute swaps (A had X, now has Y)
- Nested binding (A's friend is B who has X)
- Temporal binding (before/after states)
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
    random_number,
    PERSON_NAMES,
    COLORS,
    COMMON_NOUNS,
    ANIMALS,
)


@dataclass
class TwoEntityBindingTemplate(TestTemplate):
    """
    Basic two-entity binding test.

    Pattern: A has X, B has Y. What does A have?
    """

    category = "binding_tests"
    subcategory = "two_entity"

    def __init__(self, difficulty: Difficulty, query_position: TargetPosition):
        self.difficulty = difficulty
        self.query_position = query_position
        self.template_id = f"binding_two_{difficulty.name.lower()}_{query_position.name.lower()}"

    def generate(self, rng: random.Random) -> TestCase:
        # Select entities and attributes
        names = rng.sample(PERSON_NAMES, 2)
        colors = rng.sample(COLORS, 2)

        entity_a, entity_b = names
        attr_a, attr_b = colors

        # Query for first or second entity based on position
        if self.query_position == TargetPosition.START:
            query_entity = entity_a
            answer = attr_a
        else:
            query_entity = entity_b
            answer = attr_b

        # Vary the format
        formats = [
            f"""{entity_a}'s color is {attr_a}.
{entity_b}'s color is {attr_b}.
What is {query_entity}'s color?""",

            f"""Color assignment:
- {entity_a}: {attr_a}
- {entity_b}: {attr_b}
{query_entity}'s color is""",

            f"""{entity_a} has a {attr_a} ball.
{entity_b} has a {attr_b} ball.
What color is {query_entity}'s ball?""",
        ]

        prompt = rng.choice(formats)
        choices = [f" {attr_a}", f" {attr_b}"]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {answer}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=self.query_position,
        )


@dataclass
class ThreeEntityBindingTemplate(TestTemplate):
    """
    Three-entity binding with more interference.
    """

    category = "binding_tests"
    subcategory = "three_entity"

    def __init__(self, difficulty: Difficulty, query_position: TargetPosition):
        self.difficulty = difficulty
        self.query_position = query_position
        self.template_id = f"binding_three_{difficulty.name.lower()}_{query_position.name.lower()}"

    def generate(self, rng: random.Random) -> TestCase:
        names = rng.sample(PERSON_NAMES, 3)
        values = [random_number(rng, 10, 99) for _ in range(3)]

        # Determine which entity to query
        pos_map = {
            TargetPosition.START: 0,
            TargetPosition.MIDDLE: 1,
            TargetPosition.END: 2,
        }
        query_idx = pos_map[self.query_position]
        query_entity = names[query_idx]
        answer = values[query_idx]

        prompt = f"""{names[0]}'s score is {values[0]}.
{names[1]}'s score is {values[1]}.
{names[2]}'s score is {values[2]}.
What is {query_entity}'s score?"""

        choices = [f" {v}" for v in values]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {answer}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=self.query_position,
        )


@dataclass
class AttributeSwapTemplate(TestTemplate):
    """
    Test tracking attribute changes.

    Pattern: A had X, now has Y. What does A have?
    """

    category = "binding_tests"
    subcategory = "attribute_swap"

    def __init__(self, difficulty: Difficulty):
        self.difficulty = difficulty
        self.template_id = f"binding_swap_{difficulty.name.lower()}"

    def generate(self, rng: random.Random) -> TestCase:
        name = rng.choice(PERSON_NAMES)
        colors = rng.sample(COLORS, 2)
        old_color, new_color = colors

        # Different phrasings of the swap
        formats = [
            f"""{name}'s favorite color was {old_color}.
{name}'s favorite color is now {new_color}.
What is {name}'s current favorite color?""",

            f"""Before: {name} liked {old_color}.
After: {name} likes {new_color}.
What does {name} like now?""",

            f"""{name} had a {old_color} car.
{name} got a new {new_color} car.
What color is {name}'s current car?""",
        ]

        prompt = rng.choice(formats)
        choices = [f" {new_color}", f" {old_color}"]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {new_color}",  # Should be the NEW value
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
        )


@dataclass
class NestedBindingTemplate(TestTemplate):
    """
    Nested binding through relationships.

    Pattern: A's friend is B. B has X. What does A's friend have?
    """

    category = "binding_tests"
    subcategory = "nested"

    def __init__(self, difficulty: Difficulty):
        self.difficulty = difficulty
        self.template_id = f"binding_nested_{difficulty.name.lower()}"

    def generate(self, rng: random.Random) -> TestCase:
        names = rng.sample(PERSON_NAMES, 3)
        items = rng.sample(ANIMALS, 2)

        entity_a = names[0]
        entity_b = names[1]
        entity_c = names[2]  # Distractor
        item_b = items[0]
        item_c = items[1]

        # Different relationship types
        relationships = ["friend", "neighbor", "sibling", "colleague"]
        relationship = rng.choice(relationships)

        prompt = f"""{entity_a}'s {relationship} is {entity_b}.
{entity_b} has a {item_b}.
{entity_c} has a {item_c}.
What does {entity_a}'s {relationship} have?"""

        choices = [f" {item_b}", f" {item_c}"]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {item_b}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
        )


@dataclass
class TemporalBindingTemplate(TestTemplate):
    """
    Temporal binding with before/after states.

    Pattern: Initially A had X. Then Y happened. Now A has Z.
    """

    category = "binding_tests"
    subcategory = "temporal"

    def __init__(self, difficulty: Difficulty, query_time: str = "after"):
        self.difficulty = difficulty
        self.query_time = query_time  # "before" or "after"
        self.template_id = f"binding_temporal_{difficulty.name.lower()}_{query_time}"

    def generate(self, rng: random.Random) -> TestCase:
        name = rng.choice(PERSON_NAMES)
        values = [random_number(rng, 10, 50), random_number(rng, 51, 99)]
        before_val, after_val = values

        prompt = f"""At 9 AM, {name} had ${before_val}.
{name} worked and earned money.
At 5 PM, {name} had ${after_val}.
How much money did {name} have at """

        if self.query_time == "before":
            prompt += "9 AM?"
            answer = before_val
        else:
            prompt += "5 PM?"
            answer = after_val

        choices = [f" ${before_val}", f" ${after_val}"]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" ${answer}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
        )


@dataclass
class MultiAttributeBindingTemplate(TestTemplate):
    """
    Multiple attributes per entity.

    Pattern: A is tall and has blue eyes. B is short and has green eyes.
    """

    category = "binding_tests"
    subcategory = "multi_attribute"

    def __init__(self, difficulty: Difficulty):
        self.difficulty = difficulty
        self.template_id = f"binding_multi_{difficulty.name.lower()}"

    def generate(self, rng: random.Random) -> TestCase:
        names = rng.sample(PERSON_NAMES, 2)
        sizes = ["tall", "short"]
        colors = rng.sample(COLORS, 2)

        rng.shuffle(sizes)

        # Assign attributes
        entity_a, entity_b = names
        size_a, size_b = sizes
        color_a, color_b = colors

        # Query about one attribute of one entity
        query_entity = rng.choice([entity_a, entity_b])
        query_attribute = rng.choice(["size", "color"])

        if query_entity == entity_a:
            answer = size_a if query_attribute == "size" else color_a
            wrong = size_b if query_attribute == "size" else color_b
        else:
            answer = size_b if query_attribute == "size" else color_b
            wrong = size_a if query_attribute == "size" else color_a

        prompt = f"""{entity_a} is {size_a} and has {color_a} eyes.
{entity_b} is {size_b} and has {color_b} eyes.
"""

        if query_attribute == "size":
            prompt += f"Is {query_entity} tall or short?"
        else:
            prompt += f"What color are {query_entity}'s eyes?"

        choices = [f" {answer}", f" {wrong}"]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {answer}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
        )


def get_all_binding_templates() -> list[TestTemplate]:
    """Get all binding test templates."""
    templates = []

    for difficulty in Difficulty:
        for position in TargetPosition:
            templates.append(TwoEntityBindingTemplate(difficulty, position))
            templates.append(ThreeEntityBindingTemplate(difficulty, position))

        templates.append(AttributeSwapTemplate(difficulty))
        templates.append(NestedBindingTemplate(difficulty))
        templates.append(TemporalBindingTemplate(difficulty, "before"))
        templates.append(TemporalBindingTemplate(difficulty, "after"))
        templates.append(MultiAttributeBindingTemplate(difficulty))

    return templates


# Export TEMPLATES for consistency with other template modules
TEMPLATES = get_all_binding_templates()
