"""
World knowledge templates.

Tests factual recall about geography, science, history, and general knowledge.
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


class CapitalsTemplate(TestTemplate):
    """Country capitals."""

    category = "world_knowledge"
    subcategory = "capitals"

    CAPITALS = {
        # Easy - very common
        "France": "Paris",
        "Japan": "Tokyo",
        "Italy": "Rome",
        "Germany": "Berlin",
        "Spain": "Madrid",
        "United Kingdom": "London",
        "China": "Beijing",
        "Russia": "Moscow",
        # Medium
        "Brazil": "Brasilia",
        "Australia": "Canberra",
        "Canada": "Ottawa",
        "India": "New Delhi",
        "South Korea": "Seoul",
        "Mexico": "Mexico City",
        # Hard
        "Myanmar": "Naypyidaw",
        "Sri Lanka": "Sri Jayawardenepura Kotte",
        "Kazakhstan": "Astana",
        "Nigeria": "Abuja",
        "Turkey": "Ankara",
    }

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))

        if difficulty == Difficulty.EASY:
            countries = ["France", "Japan", "Italy", "Germany", "Spain", "United Kingdom", "China", "Russia"]
        elif difficulty == Difficulty.MEDIUM:
            countries = ["Brazil", "Australia", "Canada", "India", "South Korea", "Mexico"]
        else:
            countries = ["Myanmar", "Sri Lanka", "Kazakhstan", "Nigeria", "Turkey"]

        country = rng.choice(countries)
        capital = self.CAPITALS[country]

        formats = [
            f"What is the capital of {country}?",
            f"Capital of {country}:",
            f"The capital city of {country} is",
        ]
        prompt = rng.choice(formats)
        expected = capital

        return TestCase(
            id=f"fact_capital_{country.lower().replace(' ', '_')}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"country": country, "type": "capital"},
        )


class ScienceFactsTemplate(TestTemplate):
    """Basic science facts."""

    category = "world_knowledge"
    subcategory = "science"

    FACTS = {
        Difficulty.EASY: [
            ("What is the chemical formula for water?", "H2O"),
            ("How many planets are in our solar system?", "8"),
            ("What is the largest organ in the human body?", "skin"),
            ("What gas do plants absorb from the air?", "carbon dioxide"),
            ("What is the boiling point of water in Celsius?", "100"),
        ],
        Difficulty.MEDIUM: [
            ("What is the speed of light in km/s (approximately)?", "300000"),
            ("What is the atomic number of carbon?", "6"),
            ("How many bones are in the adult human body?", "206"),
            ("What is the chemical symbol for gold?", "Au"),
            ("What is the largest planet in our solar system?", "Jupiter"),
        ],
        Difficulty.HARD: [
            ("What is the half-life of Carbon-14 in years (approximately)?", "5730"),
            ("What is the Avogadro constant (order of magnitude)?", "10^23"),
            ("What is the atomic mass of oxygen?", "16"),
            ("What is the closest star to Earth besides the Sun?", "Proxima Centauri"),
            ("What is the pH of pure water?", "7"),
        ],
    }

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        question, answer = rng.choice(self.FACTS[difficulty])

        return TestCase(
            id=f"fact_science_{answer[:10].replace(' ', '_')}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=question,
            expected=answer,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "science"},
        )


class DateFactsTemplate(TestTemplate):
    """Historical dates and calendar facts."""

    category = "world_knowledge"
    subcategory = "dates"

    FACTS = {
        Difficulty.EASY: [
            ("How many days are in a week?", "7"),
            ("How many months are in a year?", "12"),
            ("How many days are in a non-leap year?", "365"),
            ("What month comes after March?", "April"),
            ("How many hours are in a day?", "24"),
        ],
        Difficulty.MEDIUM: [
            ("How many days are in February in a leap year?", "29"),
            ("What year did World War II end?", "1945"),
            ("How many seconds are in an hour?", "3600"),
            ("What century are we in?", "21st"),
            ("How many weeks are in a year (approximately)?", "52"),
        ],
        Difficulty.HARD: [
            ("What year was the Declaration of Independence signed?", "1776"),
            ("How many days are in a leap year?", "366"),
            ("What year did the Berlin Wall fall?", "1989"),
            ("In what year did humans first land on the moon?", "1969"),
            ("What year did the French Revolution begin?", "1789"),
        ],
    }

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        question, answer = rng.choice(self.FACTS[difficulty])

        return TestCase(
            id=f"fact_date_{answer[:10]}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=question,
            expected=answer,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "dates"},
        )


class GeographyTemplate(TestTemplate):
    """Geography facts."""

    category = "world_knowledge"
    subcategory = "geography"

    FACTS = {
        Difficulty.EASY: [
            ("What is the largest ocean?", "Pacific"),
            ("What is the longest river in the world?", "Nile"),
            ("What continent is Egypt in?", "Africa"),
            ("What is the largest country by area?", "Russia"),
            ("How many continents are there?", "7"),
        ],
        Difficulty.MEDIUM: [
            ("What is the highest mountain in the world?", "Everest"),
            ("What is the smallest country in the world?", "Vatican City"),
            ("What ocean is between Europe and America?", "Atlantic"),
            ("What is the largest desert in the world?", "Sahara"),
            ("What country has the largest population?", "China"),
        ],
        Difficulty.HARD: [
            ("What is the deepest ocean trench?", "Mariana Trench"),
            ("What is the largest lake by surface area?", "Caspian Sea"),
            ("What river flows through Paris?", "Seine"),
            ("What mountain range separates Europe from Asia?", "Urals"),
            ("What is the driest continent?", "Antarctica"),
        ],
    }

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        question, answer = rng.choice(self.FACTS[difficulty])

        return TestCase(
            id=f"fact_geo_{answer[:10].replace(' ', '_')}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=question,
            expected=answer,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "geography"},
        )


class LanguageFactsTemplate(TestTemplate):
    """Language and linguistics facts."""

    category = "world_knowledge"
    subcategory = "language"

    FACTS = {
        Difficulty.EASY: [
            ("How many letters are in the English alphabet?", "26"),
            ("What is 'hello' in Spanish?", "hola"),
            ("What language is spoken in Brazil?", "Portuguese"),
            ("How many vowels are in English?", "5"),
            ("What language is spoken in France?", "French"),
        ],
        Difficulty.MEDIUM: [
            ("What is the most spoken language in the world?", "Mandarin"),
            ("What is 'thank you' in Japanese?", "arigatou"),
            ("What language uses Cyrillic script?", "Russian"),
            ("What is 'goodbye' in German?", "auf Wiedersehen"),
            ("What is the official language of Egypt?", "Arabic"),
        ],
        Difficulty.HARD: [
            ("What language family does Finnish belong to?", "Uralic"),
            ("What is the oldest known written language?", "Sumerian"),
            ("How many tones does Mandarin Chinese have?", "4"),
            ("What writing system does Japanese use alongside kanji?", "hiragana"),
            ("What language is spoken in Iceland?", "Icelandic"),
        ],
    }

    def generate(self, rng: random.Random) -> TestCase:
        difficulty = rng.choice(list(Difficulty))
        question, answer = rng.choice(self.FACTS[difficulty])

        return TestCase(
            id=f"fact_lang_{answer[:10].replace(' ', '_')}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=difficulty,
            prompt=question,
            expected=answer,
            kind=EvalKind.GENERATION,
            target_position=TargetPosition.END,
            metadata={"type": "language"},
        )


# Export all templates
TEMPLATES = [
    CapitalsTemplate(),
    ScienceFactsTemplate(),
    DateFactsTemplate(),
    GeographyTemplate(),
    LanguageFactsTemplate(),
]
