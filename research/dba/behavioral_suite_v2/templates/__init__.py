"""
Template definitions for behavioral test generation.

Each module defines templates for a specific category of tests.
All 18 categories are covered:
- copy_tasks: Raw memorization fidelity
- fewshot_learning: In-context pattern recognition
- distractor_tests: Attention focus under interference
- reasoning: Logical inference
- arithmetic: Numerical reasoning
- sequences: Pattern extrapolation
- world_knowledge: Factual recall
- semantic: Meaning comprehension
- format_preservation: Structural fidelity
- long_context: Context window utilization
- robustness: Consistency under variation
- edge_cases: Boundary conditions
- attention_probes: High/low-rank attention diagnostics
- instruction_following: Instruction adherence
- consistency_checks: Same capability, different format
- adversarial: Prompt injection resistance
- binding_tests: Entity-attribute binding
- multi_hop: Multi-step reasoning chains
"""
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

# Import all template modules
from . import copy_tasks
from . import fewshot_learning
from . import distractor_tests
from . import reasoning
from . import arithmetic
from . import sequences
from . import world_knowledge
from . import semantic
from . import format_preservation
from . import long_context
from . import robustness
from . import edge_cases
from . import attention_probes
from . import instruction_following
from . import consistency_checks
from . import adversarial
from . import binding_tests
from . import multi_hop

# Collect all templates by category
ALL_TEMPLATES = {
    "copy_tasks": copy_tasks.TEMPLATES,
    "fewshot_learning": fewshot_learning.TEMPLATES,
    "distractor_tests": distractor_tests.TEMPLATES,
    "reasoning": reasoning.TEMPLATES,
    "arithmetic": arithmetic.TEMPLATES,
    "sequences": sequences.TEMPLATES,
    "world_knowledge": world_knowledge.TEMPLATES,
    "semantic": semantic.TEMPLATES,
    "format_preservation": format_preservation.TEMPLATES,
    "long_context": long_context.TEMPLATES,
    "robustness": robustness.TEMPLATES,
    "edge_cases": edge_cases.TEMPLATES,
    "attention_probes": attention_probes.TEMPLATES,
    "instruction_following": instruction_following.TEMPLATES,
    "consistency_checks": consistency_checks.TEMPLATES,
    "adversarial": adversarial.TEMPLATES,
    "binding_tests": binding_tests.TEMPLATES,
    "multi_hop": multi_hop.TEMPLATES,
}

__all__ = [
    # Base types
    "TestCase",
    "TestTemplate",
    "Difficulty",
    "TargetPosition",
    "EvalKind",
    "random_position",
    "random_word",
    "random_words",
    # Template collection
    "ALL_TEMPLATES",
    # Individual modules
    "copy_tasks",
    "fewshot_learning",
    "distractor_tests",
    "reasoning",
    "arithmetic",
    "sequences",
    "world_knowledge",
    "semantic",
    "format_preservation",
    "long_context",
    "robustness",
    "edge_cases",
    "attention_probes",
    "instruction_following",
    "consistency_checks",
    "adversarial",
    "binding_tests",
    "multi_hop",
]
