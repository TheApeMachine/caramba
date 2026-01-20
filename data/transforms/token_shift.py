"""Token shift transform

Splits token sequences into input/target pairs for next-token prediction,
creating the standard format for autoregressive language modeling.
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from runtime.tensordict_utils import TensorDictBase, as_tensordict


@dataclass(frozen=True, slots=True)
class TokenShift:
    """Create input/target pairs from token sequence

    Splits a token sequence into input and target pairs for next-token
    prediction, where target is the input shifted by one position.
    """
    src_key: str
    input_key: str
    target_key: str

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Generate input-target pairs

        Takes a sequence of tokens and creates (input, target) pairs where each
        target token is the next token after the corresponding input position,
        which is the standard format for autoregressive language modeling.
        """
        d = dict(td)
        tok = d.get(self.src_key, None)
        if not isinstance(tok, Tensor):
            return as_tensordict(d)
        if tok.dim() < 1 or tok.size(-1) < 2:
            raise ValueError(f"token_shift expects {self.src_key} with last dim >= 2")
        d[self.input_key] = tok[..., :-1]
        d[self.target_key] = tok[..., 1:]
        return as_tensordict(d)
