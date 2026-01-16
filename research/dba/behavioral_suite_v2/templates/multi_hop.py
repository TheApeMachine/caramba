"""
Multi-hop reasoning test templates.

Tests the model's ability to chain multiple pieces of information:
- Transitive chains (A > B, B > C => A > C)
- Fact chains (combining multiple facts)
- Arithmetic chains (multi-step calculations)
- Reference chains (X has what Y has, Y has what Z has)
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
    ANIMALS,
)


@dataclass
class TransitiveChainTemplate(TestTemplate):
    """
    Transitive reasoning chains.

    Pattern: A > B, B > C, C > D. Is A > D?
    """

    category = "multi_hop"
    subcategory = "transitive"

    def __init__(self, difficulty: Difficulty, chain_length: int = 3):
        self.difficulty = difficulty
        self.chain_length = chain_length
        self.template_id = f"multihop_trans_{difficulty.name.lower()}_{chain_length}hop"

    def generate(self, rng: random.Random) -> TestCase:
        # Generate chain of relations
        items = [chr(ord('A') + i) for i in range(self.chain_length + 1)]
        relations = rng.choice([">", "is bigger than", "is taller than", "is older than"])

        # Build premise statements
        premises = []
        for i in range(self.chain_length):
            premises.append(f"{items[i]} {relations} {items[i+1]}")

        # Query about transitive relation
        query_a = items[0]
        query_b = items[-1]

        # Sometimes ask about valid inference, sometimes invalid
        if rng.random() < 0.7:
            # Valid: first > last
            answer = "Yes"
            question = f"Is {query_a} {relations.replace('is ', '')} {query_b}?"
        else:
            # Invalid: last > first (reverse)
            answer = "No"
            question = f"Is {query_b} {relations.replace('is ', '')} {query_a}?"

        prompt = "\n".join(premises) + "\n" + question

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {answer}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=[" Yes", " No"],
        )


@dataclass
class FactChainTemplate(TestTemplate):
    """
    Combining multiple facts to answer a question.

    Pattern: A lives in X. X is in country Y. What country does A live in?
    """

    category = "multi_hop"
    subcategory = "fact_chain"

    def __init__(self, difficulty: Difficulty, num_hops: int = 2):
        self.difficulty = difficulty
        self.num_hops = num_hops
        self.template_id = f"multihop_fact_{difficulty.name.lower()}_{num_hops}hop"

    def generate(self, rng: random.Random) -> TestCase:
        names = rng.sample(PERSON_NAMES, 2)
        cities = ["Paris", "Tokyo", "London", "Berlin"]
        countries = ["France", "Japan", "UK", "Germany"]

        person = names[0]
        city_idx = rng.randint(0, len(cities) - 1)
        city = cities[city_idx]
        country = countries[city_idx]

        # Add distractors
        other_person = names[1]
        other_city_idx = (city_idx + 1) % len(cities)
        other_city = cities[other_city_idx]
        other_country = countries[other_city_idx]

        if self.num_hops == 2:
            prompt = f"""{person} lives in {city}.
{other_person} lives in {other_city}.
{city} is in {country}.
{other_city} is in {other_country}.
What country does {person} live in?"""
        else:  # 3 hops
            continent = "Europe" if country in ["France", "UK", "Germany"] else "Asia"
            prompt = f"""{person} lives in {city}.
{city} is in {country}.
{country} is in {continent}.
What continent does {person} live in?"""
            country = continent  # Update answer

        choices = [f" {country}", f" {other_country}"]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {country}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
        )


@dataclass
class ArithmeticChainTemplate(TestTemplate):
    """
    Multi-step arithmetic calculations.

    Pattern: Start with X. Add Y. Multiply by Z. What is the result?
    """

    category = "multi_hop"
    subcategory = "arithmetic_chain"

    def __init__(self, difficulty: Difficulty, num_operations: int = 2):
        self.difficulty = difficulty
        self.num_operations = num_operations
        self.template_id = f"multihop_arith_{difficulty.name.lower()}_{num_operations}op"

    def generate(self, rng: random.Random) -> TestCase:
        # Start with a base number
        base_ranges = {
            Difficulty.EASY: (5, 20),
            Difficulty.MEDIUM: (10, 50),
            Difficulty.HARD: (20, 100),
        }
        min_base, max_base = base_ranges[self.difficulty]
        current = random_number(rng, min_base, max_base)

        operations = []
        for _ in range(self.num_operations):
            op = rng.choice(["add", "subtract", "double"])
            if op == "add":
                amount = random_number(rng, 1, 10)
                operations.append(f"Add {amount}")
                current += amount
            elif op == "subtract":
                amount = random_number(rng, 1, min(10, current - 1))
                operations.append(f"Subtract {amount}")
                current -= amount
            else:  # double
                operations.append("Double it")
                current *= 2

        # Build prompt
        prompt_lines = [f"Start with {current - sum(range(self.num_operations))}"]  # This is wrong, need to track
        # Actually rebuild properly
        start = random_number(rng, min_base, max_base)
        current = start
        operations = []
        for _ in range(self.num_operations):
            op = rng.choice(["add", "subtract"])
            amount = random_number(rng, 1, 10)
            if op == "add":
                operations.append(f"Add {amount}.")
                current += amount
            else:
                if current > amount:
                    operations.append(f"Subtract {amount}.")
                    current -= amount
                else:
                    operations.append(f"Add {amount}.")
                    current += amount

        prompt = f"""Start: {start}
{chr(10).join(operations)}
Final result:"""

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=current,
            kind=EvalKind.INT_GREEDY,
        )


@dataclass
class ReferenceChainTemplate(TestTemplate):
    """
    Reference chains where entities refer to each other.

    Pattern: A has what B has. B has what C has. C has X. What does A have?
    """

    category = "multi_hop"
    subcategory = "reference_chain"

    def __init__(self, difficulty: Difficulty, chain_length: int = 2):
        self.difficulty = difficulty
        self.chain_length = chain_length
        self.template_id = f"multihop_ref_{difficulty.name.lower()}_{chain_length}hop"

    def generate(self, rng: random.Random) -> TestCase:
        names = rng.sample(PERSON_NAMES, self.chain_length + 2)  # +1 for chain, +1 for distractor

        # The actual answer
        target_item = rng.choice(ANIMALS)
        distractor_item = rng.choice([a for a in ANIMALS if a != target_item])

        # Build reference chain
        chain_names = names[:self.chain_length + 1]
        distractor_name = names[self.chain_length + 1]

        statements = []
        for i in range(self.chain_length):
            statements.append(f"{chain_names[i]} has what {chain_names[i+1]} has.")

        # Final entity has the actual item
        statements.append(f"{chain_names[-1]} has a {target_item}.")
        statements.append(f"{distractor_name} has a {distractor_item}.")

        # Shuffle statements (except keep question at end)
        rng.shuffle(statements)

        prompt = "\n".join(statements) + f"\nWhat does {chain_names[0]} have?"

        choices = [f" {target_item}", f" {distractor_item}"]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {target_item}",
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
        )


@dataclass
class ComparativeChainTemplate(TestTemplate):
    """
    Comparative reasoning with multiple entities.

    Pattern: A is bigger than B. C is smaller than B. Who is biggest?
    """

    category = "multi_hop"
    subcategory = "comparative"

    def __init__(self, difficulty: Difficulty):
        self.difficulty = difficulty
        self.template_id = f"multihop_comp_{difficulty.name.lower()}"

    def generate(self, rng: random.Random) -> TestCase:
        names = rng.sample(PERSON_NAMES, 3)
        attribute = rng.choice(["taller", "older", "faster", "stronger"])
        opposite = {"taller": "shorter", "older": "younger", "faster": "slower", "stronger": "weaker"}[attribute]
        superlative = {"taller": "tallest", "older": "oldest", "faster": "fastest", "stronger": "strongest"}[attribute]

        # Create ordering: names[0] > names[1] > names[2]
        ordered = names.copy()

        # Generate statements that imply this ordering
        statements = [
            f"{ordered[0]} is {attribute} than {ordered[1]}.",
            f"{ordered[1]} is {attribute} than {ordered[2]}.",
        ]
        rng.shuffle(statements)

        # Ask for superlative
        prompt = "\n".join(statements) + f"\nWho is the {superlative}?"

        choices = [f" {n}" for n in names]

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=prompt,
            expected=f" {ordered[0]}",  # First in ordering is superlative
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
        )


def get_all_multihop_templates() -> list[TestTemplate]:
    """Get all multi-hop reasoning templates."""
    templates = []

    for difficulty in Difficulty:
        # Transitive chains of different lengths
        for length in [2, 3, 4]:
            templates.append(TransitiveChainTemplate(difficulty, length))

        # Fact chains
        for hops in [2, 3]:
            templates.append(FactChainTemplate(difficulty, hops))

        # Arithmetic chains
        for ops in [2, 3]:
            templates.append(ArithmeticChainTemplate(difficulty, ops))

        # Reference chains
        for length in [2, 3]:
            templates.append(ReferenceChainTemplate(difficulty, length))

        # Comparative
        templates.append(ComparativeChainTemplate(difficulty))

    return templates


# Export TEMPLATES for consistency with other template modules
TEMPLATES = get_all_multihop_templates()
