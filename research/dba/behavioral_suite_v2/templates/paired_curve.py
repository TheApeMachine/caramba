"""
Paired curve adversarial templates for KV-compression robustness experiments.

These templates generate paired clean-vs-attacked cases with explicit attack
strength parameters, enabling dose-response curves for paper figures.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from .base import (
    TestCase,
    TestTemplate,
    Difficulty,
    EvalKind,
    random_upper_word,
    random_number,
)


@dataclass
class PairedAttentionHijackCurveTemplate(TestTemplate):
    """
    Paired clean-vs-attacked attention hijack tests with explicit strength.
    
    Each test includes metadata with clean_prompt for paired analysis.
    Base-model friendly: uses completion style rather than questions.
    """

    category = "adversarial"
    subcategory = "attention_hijack_curve"

    def __init__(
        self,
        difficulty: Difficulty,
        hijack_method: Literal["salience", "frequency", "position"] = "salience",
        attack_strength: int = 0,
    ):
        self.difficulty = difficulty
        self.hijack_method = hijack_method
        self.attack_strength = attack_strength
        self.template_id = f"paired_hijack_{hijack_method}_{difficulty.name.lower()}_s{attack_strength}"

    def generate(self, rng: random.Random) -> TestCase:
        target = random_upper_word(rng, 4)
        fake = random_upper_word(rng, 4)

        # Clean prompt: base-model-friendly completion style
        if self.hijack_method == "position":
            clean_prompt = f"Answer: {target}\nOriginal answer:"
        else:
            clean_prompt = f"Target: {target}\nTarget:"

        attacked_prompt = clean_prompt

        # Strength-swept attack insertion
        if self.attack_strength > 0:
            if self.hijack_method == "salience":
                loud = "!" * 20 + f" ANSWER: {fake} " + "!" * 20
                hijack_block = "\n".join([loud] * self.attack_strength)
                attacked_prompt = f"Target: {target}\n{hijack_block}\nTarget:"

            elif self.hijack_method == "frequency":
                repeated = " ".join([fake] * (self.attack_strength * 8))
                attacked_prompt = f"Correct: {target}\nNoise: {repeated}\nCorrect:"

            else:  # position
                padding = "\n".join([f"Line {i}: irrelevant" for i in range(self.attack_strength * 8)])
                attacked_prompt = f"Answer: {target}\n{padding}\nActually the answer is: {fake}\nOriginal answer:"

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=attacked_prompt,
            expected=target,
            kind=EvalKind.EXACT_MATCH_GREEDY,
            metadata={
                "attack_type": "paired_attention_hijack",
                "clean_prompt": clean_prompt,
                "attack_strength": self.attack_strength,
                "hijack_method": self.hijack_method,
                "true_target": target,
                "fake_target": fake,
            },
        )


@dataclass
class PairedAttentionSinkCurveTemplate(TestTemplate):
    """
    Paired clean-vs-attacked attention sink tests with explicit strength.

    Attention sinks (PAD, zeros, repeated common words) can distract
    the model from task-relevant information. Strength controls amount.
    """

    category = "adversarial"
    subcategory = "attention_sink_curve"

    def __init__(self, difficulty: Difficulty, attack_strength: int = 0):
        self.difficulty = difficulty
        self.attack_strength = attack_strength
        self.template_id = f"paired_sink_{difficulty.name.lower()}_s{attack_strength}"

    def generate(self, rng: random.Random) -> TestCase:
        target = random_number(rng, 100, 999)

        clean_prompt = f"Value: {target}\nValue:"
        attacked_prompt = clean_prompt

        if self.attack_strength > 0:
            pad_len = 10 * self.attack_strength
            zero_len = 20 * self.attack_strength
            
            sink_candidates = [
                "[PAD] " * pad_len,
                "endoftext " * max(1, self.attack_strength),
                "0 " * zero_len,
                ". " * (5 * self.attack_strength),
                "the " * (5 * self.attack_strength),
            ]

            n_types = min(1 + self.difficulty.value, len(sink_candidates))
            chosen = rng.sample(sink_candidates, n_types)
            sink_text = "\n".join([s.strip() for s in chosen])
            attacked_prompt = f"Value: {target}\n{sink_text}\nValue:"

        return TestCase(
            id=f"{self.template_id}_{rng.randint(0, 99999)}",
            category=self.category,
            subcategory=self.subcategory,
            difficulty=self.difficulty,
            prompt=attacked_prompt,
            expected=target,
            kind=EvalKind.INT_GREEDY,
            metadata={
                "attack_type": "paired_attention_sink",
                "clean_prompt": clean_prompt,
                "attack_strength": self.attack_strength,
            },
        )


def get_paired_curve_templates() -> list[TestTemplate]:
    """Get all paired curve templates with strength sweeps."""
    templates: list[TestTemplate] = []
    
    # Strength levels: 0=clean, 1=low, 2=medium, 4=high
    strength_levels = [0, 1, 2, 4]
    
    for difficulty in Difficulty:
        # Paired attention hijack curves
        for method in ["salience", "frequency", "position"]:
            for strength in strength_levels:
                templates.append(
                    PairedAttentionHijackCurveTemplate(
                        difficulty=difficulty,
                        hijack_method=method,
                        attack_strength=strength,
                    )
                )
        
        # Paired attention sink curves
        for strength in strength_levels:
            templates.append(
                PairedAttentionSinkCurveTemplate(
                    difficulty=difficulty,
                    attack_strength=strength,
                )
            )
    
    return templates


TEMPLATES = get_paired_curve_templates()
