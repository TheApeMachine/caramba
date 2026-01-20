"""Evaluator package

Implements objective-truth gates and scoring for CCP-style workflows:
- validity checks (frames, schemas)
- policy enforcement (capabilities, budgets)
- scoring (ground-truth correctness)
- regression suite runner (non-regression on fixed traces)
"""

from evaluator.policy import PolicyGate
from evaluator.regression import RegressionSuite
from evaluator.score import Scorecard
from evaluator.validity import ValidityGate

__all__ = [
    "PolicyGate",
    "RegressionSuite",
    "Scorecard",
    "ValidityGate",
]

