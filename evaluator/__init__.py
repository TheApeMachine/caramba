"""Evaluator package

Implements objective-truth gates and scoring for CCP-style workflows:
- validity checks (frames, schemas)
- policy enforcement (capabilities, budgets)
- scoring (ground-truth correctness)
- regression suite runner (non-regression on fixed traces)
"""

from caramba.evaluator.policy import PolicyGate
from caramba.evaluator.regression import RegressionSuite
from caramba.evaluator.score import Scorecard
from caramba.evaluator.validity import ValidityGate

__all__ = [
    "PolicyGate",
    "RegressionSuite",
    "Scorecard",
    "ValidityGate",
]

