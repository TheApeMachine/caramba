"""
Semantic understanding templates.

Tests meaning comprehension including synonyms, antonyms, analogies,
and category membership.

NOTE: These templates use CHOICE_LOGPROB scoring for clean capability measurement.
Each test provides MCQ-style choices to eliminate decoder confounds.
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


class SynonymTemplate(TestTemplate):
    """Find synonyms for words - MCQ format."""

    category = "semantic"
    subcategory = "synonym"

    SYNONYMS = {
        Difficulty.EASY: [
            ("happy", ["glad", "joyful", "pleased", "content"]),
            ("big", ["large", "huge", "enormous", "giant"]),
            ("fast", ["quick", "rapid", "swift", "speedy"]),
            ("smart", ["intelligent", "clever", "bright", "wise"]),
            ("pretty", ["beautiful", "lovely", "attractive", "gorgeous"]),
        ],
        Difficulty.MEDIUM: [
            ("angry", ["furious", "irate", "enraged", "livid"]),
            ("sad", ["melancholy", "sorrowful", "dejected", "gloomy"]),
            ("brave", ["courageous", "valiant", "fearless", "bold"]),
            ("quiet", ["silent", "hushed", "muted", "tranquil"]),
            ("old", ["ancient", "aged", "elderly", "antique"]),
        ],
        Difficulty.HARD: [
            ("ephemeral", ["transient", "fleeting", "momentary", "brief"]),
            ("ubiquitous", ["omnipresent", "pervasive", "universal"]),
            ("meticulous", ["scrupulous", "thorough", "precise", "careful"]),
            ("pragmatic", ["practical", "realistic", "sensible"]),
            ("ambiguous", ["vague", "unclear", "equivocal", "obscure"]),
        ],
    }

    # Non-synonyms for distractors
    DISTRACTORS = ["terrible", "enormous", "frozen", "ancient", "silent", "rapid", "gentle", "fierce"]

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        word, synonyms = rng.choice(self.SYNONYMS[difficulty])
        correct = rng.choice(synonyms)

        # Build choices: correct answer + distractors
        distractors = [d for d in self.DISTRACTORS if d not in synonyms and d != word]
        selected_distractors = rng.sample(distractors, min(3, len(distractors)))

        choices = [f" {correct}"] + [f" {d}" for d in selected_distractors]
        rng.shuffle(choices)

        prompt = f"What is a synonym for '{word}'? Options: {', '.join(c.strip() for c in choices)}\nAnswer:"
        expected = f" {correct}"

        return TestCase(
            id=f"semantic_syn_{word}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"word": word, "valid_answers": synonyms, "type": "synonym"},
        )


class AntonymTemplate(TestTemplate):
    """Find antonyms for words - MCQ format."""

    category = "semantic"
    subcategory = "antonym"

    ANTONYMS = {
        Difficulty.EASY: [
            ("hot", ["cold", "cool", "freezing"]),
            ("up", ["down"]),
            ("big", ["small", "little", "tiny"]),
            ("fast", ["slow"]),
            ("happy", ["sad", "unhappy"]),
            ("light", ["dark", "heavy"]),
            ("old", ["new", "young"]),
        ],
        Difficulty.MEDIUM: [
            ("ancient", ["modern", "new", "recent"]),
            ("abundant", ["scarce", "rare", "sparse"]),
            ("genuine", ["fake", "false", "artificial"]),
            ("generous", ["stingy", "selfish", "greedy"]),
            ("temporary", ["permanent", "lasting", "enduring"]),
        ],
        Difficulty.HARD: [
            ("exacerbate", ["alleviate", "mitigate", "ameliorate"]),
            ("ephemeral", ["permanent", "eternal", "everlasting"]),
            ("verbose", ["concise", "terse", "succinct"]),
            ("benevolent", ["malevolent", "malicious", "cruel"]),
            ("obscure", ["clear", "obvious", "evident"]),
        ],
    }

    # Non-antonyms for distractors
    DISTRACTORS = ["similar", "related", "equal", "moderate", "average", "typical", "normal", "usual"]

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        word, antonyms = rng.choice(self.ANTONYMS[difficulty])
        correct = rng.choice(antonyms)

        # Build choices: correct answer + distractors
        distractors = [d for d in self.DISTRACTORS if d not in antonyms and d != word]
        selected_distractors = rng.sample(distractors, min(3, len(distractors)))

        choices = [f" {correct}"] + [f" {d}" for d in selected_distractors]
        rng.shuffle(choices)

        prompt = f"What is the opposite of '{word}'? Options: {', '.join(c.strip() for c in choices)}\nAnswer:"
        expected = f" {correct}"

        return TestCase(
            id=f"semantic_ant_{word}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"word": word, "valid_answers": antonyms, "type": "antonym"},
        )


class AnalogyTemplate(TestTemplate):
    """Word analogies: A is to B as C is to ? - MCQ format."""

    category = "semantic"
    subcategory = "analogy"

    ANALOGIES = {
        Difficulty.EASY: [
            ("dog", "puppy", "cat", "kitten"),
            ("hot", "cold", "up", "down"),
            ("day", "night", "summer", "winter"),
            ("man", "woman", "boy", "girl"),
            ("bird", "fly", "fish", "swim"),
        ],
        Difficulty.MEDIUM: [
            ("author", "book", "painter", "painting"),
            ("hand", "finger", "foot", "toe"),
            ("tree", "forest", "star", "galaxy"),
            ("doctor", "hospital", "teacher", "school"),
            ("pen", "write", "knife", "cut"),
        ],
        Difficulty.HARD: [
            ("marathon", "race", "symphony", "music"),
            ("telescope", "stars", "microscope", "cells"),
            ("archaeologist", "artifacts", "astronomer", "planets"),
            ("novel", "chapter", "play", "act"),
            ("hunger", "eat", "thirst", "drink"),
        ],
    }

    DISTRACTORS = ["run", "walk", "speak", "think", "grow", "sleep", "work", "play"]

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        a, b, c, d = rng.choice(self.ANALOGIES[difficulty])

        # Build choices: correct answer + distractors
        distractors = [x for x in self.DISTRACTORS if x != d]
        selected_distractors = rng.sample(distractors, min(3, len(distractors)))

        choices = [f" {d}"] + [f" {x}" for x in selected_distractors]
        rng.shuffle(choices)

        prompt = f"{a} is to {b} as {c} is to? Options: {', '.join(c.strip() for c in choices)}\nAnswer:"
        expected = f" {d}"

        return TestCase(
            id=f"semantic_analogy_{a}_{c}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"analogy": [a, b, c, d], "type": "analogy"},
        )


class CategoryMembershipTemplate(TestTemplate):
    """Identify category membership - MCQ format."""

    category = "semantic"
    subcategory = "category"

    CATEGORIES = {
        "fruit": ["apple", "banana", "orange", "grape", "mango", "pear"],
        "vegetable": ["carrot", "broccoli", "spinach", "potato", "onion", "celery"],
        "animal": ["dog", "cat", "elephant", "lion", "tiger", "bear"],
        "color": ["red", "blue", "green", "yellow", "purple", "orange"],
        "country": ["France", "Japan", "Brazil", "Egypt", "Canada", "India"],
        "planet": ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Neptune"],
        "instrument": ["piano", "guitar", "violin", "drums", "flute", "trumpet"],
    }

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        category_name = rng.choice(list(self.CATEGORIES.keys()))
        members = self.CATEGORIES[category_name]

        if difficulty == Difficulty.EASY:
            # What category does X belong to?
            item = rng.choice(members)
            category_options = list(self.CATEGORIES.keys())
            choices = [f" {category_name}"] + [f" {c}" for c in rng.sample([c for c in category_options if c != category_name], 3)]
            rng.shuffle(choices)
            prompt = f"What category does '{item}' belong to? Options: {', '.join(c.strip() for c in choices)}\nAnswer:"
            expected = f" {category_name}"
        elif difficulty == Difficulty.MEDIUM:
            # Give an example of category X - choose from options
            correct = rng.choice(members)
            # Get items from other categories as distractors
            other_items = []
            for other_cat, other_members in self.CATEGORIES.items():
                if other_cat != category_name:
                    other_items.extend(other_members)
            distractors = rng.sample(other_items, 3)
            choices = [f" {correct}"] + [f" {d}" for d in distractors]
            rng.shuffle(choices)
            prompt = f"Which is a {category_name}? Options: {', '.join(c.strip() for c in choices)}\nAnswer:"
            expected = f" {correct}"
        else:
            # Which doesn't belong?
            items = rng.sample(members, 3)
            # Pick outlier from different category
            other_cat = rng.choice([c for c in self.CATEGORIES.keys() if c != category_name])
            outlier = rng.choice(self.CATEGORIES[other_cat])
            choices = [f" {outlier}"] + [f" {i}" for i in items]
            rng.shuffle(choices)
            prompt = f"Which word doesn't belong with the others? Options: {', '.join(c.strip() for c in choices)}\nAnswer:"
            expected = f" {outlier}"

        return TestCase(
            id=f"semantic_cat_{category_name}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"category": category_name, "type": "category"},
        )


class SentimentTemplate(TestTemplate):
    """Identify sentiment of text - MCQ format."""

    category = "semantic"
    subcategory = "sentiment"

    SENTENCES = {
        "positive": [
            "I absolutely loved this movie!",
            "This is the best day ever!",
            "What a wonderful surprise!",
            "I'm so grateful for your help.",
            "This exceeded all my expectations!",
        ],
        "negative": [
            "This was a complete waste of time.",
            "I'm very disappointed with the result.",
            "What a terrible experience.",
            "I regret buying this product.",
            "This is the worst service I've received.",
        ],
        "neutral": [
            "The meeting is scheduled for 3 PM.",
            "The document contains five pages.",
            "Water boils at 100 degrees Celsius.",
            "The store opens at 9 AM.",
            "This book has 200 pages.",
        ],
    }

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            # Clear positive or negative
            sentiment = rng.choice(["positive", "negative"])
        elif difficulty == Difficulty.MEDIUM:
            # Any sentiment
            sentiment = rng.choice(["positive", "negative", "neutral"])
        else:
            # Harder to classify (neutral or subtle)
            sentiment = rng.choice(["neutral", "positive", "negative"])

        sentence = rng.choice(self.SENTENCES[sentiment])
        choices = [" positive", " negative", " neutral"]
        prompt = f'What is the sentiment of this sentence?\n"{sentence}"\nOptions: positive, negative, neutral\nAnswer:'
        expected = f" {sentiment}"

        return TestCase(
            id=f"semantic_sentiment_{sentiment}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"sentiment": sentiment, "type": "sentiment"},
        )


class WordAssociationTemplate(TestTemplate):
    """Word association - what word is related? MCQ format."""

    category = "semantic"
    subcategory = "association"

    ASSOCIATIONS = [
        ("bread", ["butter", "toast", "bakery", "wheat"]),
        ("coffee", ["cup", "morning", "caffeine", "beans"]),
        ("ocean", ["waves", "fish", "salt", "beach"]),
        ("book", ["read", "pages", "library", "author"]),
        ("music", ["song", "melody", "concert", "notes"]),
        ("fire", ["heat", "flames", "smoke", "burn"]),
        ("rain", ["umbrella", "clouds", "wet", "storm"]),
        ("doctor", ["hospital", "medicine", "patient", "health"]),
    ]

    DISTRACTORS = ["table", "window", "carpet", "ceiling", "wall", "floor", "door", "lamp"]

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        word, associations = rng.choice(self.ASSOCIATIONS)
        correct = rng.choice(associations)

        # Build choices: correct answer + distractors
        distractors = [d for d in self.DISTRACTORS if d not in associations]
        selected_distractors = rng.sample(distractors, min(3, len(distractors)))

        choices = [f" {correct}"] + [f" {d}" for d in selected_distractors]
        rng.shuffle(choices)

        prompt = f"What word is commonly associated with '{word}'? Options: {', '.join(c.strip() for c in choices)}\nAnswer:"
        expected = f" {correct}"

        return TestCase(
            id=f"semantic_assoc_{word}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.CHOICE_LOGPROB,
            choices=choices,
            target_position=TargetPosition.END,
            metadata={"word": word, "valid_answers": associations, "type": "association"},
        )


# Export all templates
TEMPLATES = [
    SynonymTemplate(),
    AntonymTemplate(),
    AnalogyTemplate(),
    CategoryMembershipTemplate(),
    SentimentTemplate(),
    WordAssociationTemplate(),
]
