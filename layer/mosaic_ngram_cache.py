"""Optional MOSAIC n-gram cache logits mixing (PPM-lite).

This is intended as a *practical* on-device improvement for copying/continuation:
- Maintain a fixed-size direct-mapped table: (N-gram hash -> top-m next-token counts)
- At inference, add a sparse logit bias for the stored next-token distribution.

Important:
- This cache is *not* a KV cache and does not scale with context length.
- The cache lives in `ctx` (external mutable state), not in model parameters.
- For simplicity and memory safety, the implementation targets batch_size=1.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from caramba.config.layer import MosaicNGramCacheLogitsLayerConfig


@dataclass
class _NGramState:
    # Rolling hash state (uint64 in python int, masked to 64 bits).
    fp: int
    filled: int
    wpos: int
    window: list[int]  # length n, stores token_id+1 (0 means empty)

    # Direct-mapped table.
    # Store key as (fp+1) so 0 means empty.
    keys: array  # 'Q' length table_size
    toks: array  # 'I' length table_size*top_m, stores token_id+1 (0 means empty)
    cnts: array  # 'H' length table_size*top_m, counts (uint16)


def _get_ngram_state(ctx: object | None, key: str) -> _NGramState | None:
    if ctx is None:
        return None
    store = getattr(ctx, "_mosaic", None)
    if not isinstance(store, dict):
        return None
    st = store.get(key)
    return st if isinstance(st, _NGramState) else None


def _set_ngram_state(ctx: object | None, key: str, st: _NGramState) -> None:
    if ctx is None:
        return
    store = getattr(ctx, "_mosaic", None)
    if store is None:
        store = {}
        try:
            setattr(ctx, "_mosaic", store)
        except Exception:
            return
    if isinstance(store, dict):
        store[key] = st


class MosaicNGramCacheLogitsLayer(nn.Module):
    """Mix an n-gram cache distribution into logits (sparse, fixed-size table)."""

    def __init__(self, config: MosaicNGramCacheLogitsLayerConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.n = int(config.n)
        self.table_size = int(config.table_size)
        self.top_m = int(config.top_m)
        self.weight = float(config.weight)

        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.n < 1:
            raise ValueError("n must be positive")
        if self.table_size < 8:
            raise ValueError("table_size must be >= 8")
        if self.top_m < 1:
            raise ValueError("top_m must be >= 1")

        # Stable key for ctx storage.
        self._ctx_key = f"mosaic_ngram::{id(self)}"

        # Rolling hash constants (uint64-like behavior in int64).
        self._base = 1315423911  # large odd constant
        self._mask64 = (1 << 64) - 1
        # base^n mod 2^64 for rolling hash subtraction.
        self._base_pow_n = pow(int(self._base), int(self.n), 1 << 64)

    def _ensure_state(self, ctx: object | None) -> _NGramState | None:
        if ctx is None:
            return None
        st = _get_ngram_state(ctx, self._ctx_key)
        if st is not None:
            # If config changed, re-init.
            if len(st.window) == int(self.n) and len(st.keys) == int(self.table_size):
                return st

        # Allocate fresh fixed-size state (batch_size=1).
        keys = array("Q", [0]) * int(self.table_size)
        toks = array("I", [0]) * int(self.table_size * self.top_m)
        cnts = array("H", [0]) * int(self.table_size * self.top_m)
        st2 = _NGramState(
            fp=0,
            filled=0,
            wpos=0,
            window=[0] * int(self.n),
            keys=keys,
            toks=toks,
            cnts=cnts,
        )
        _set_ngram_state(ctx, self._ctx_key, st2)
        return st2

    def _iter_topm(self, st: _NGramState, slot: int) -> Iterable[tuple[int, int, int]]:
        """Yield (i, tok_plus1, count) for the slot."""
        base = int(slot) * int(self.top_m)
        for i in range(int(self.top_m)):
            tokp1 = int(st.toks[base + i])
            if tokp1 <= 0:
                continue
            yield i, tokp1, int(st.cnts[base + i])

    def _add_count(self, st: _NGramState, slot: int, tok_id: int) -> None:
        """Update top-m counts in a direct-mapped slot."""
        tokp1 = int(tok_id) + 1
        base = int(slot) * int(self.top_m)

        # Find existing token, empty slot, and min-count slot.
        empty_i = -1
        min_i = 0
        min_c = int(st.cnts[base + 0])
        for i in range(int(self.top_m)):
            tp1 = int(st.toks[base + i])
            c = int(st.cnts[base + i])
            if tp1 == tokp1:
                st.cnts[base + i] = int(min(65535, c + 1))
                return
            if tp1 == 0 and empty_i < 0:
                empty_i = i
            if c < min_c:
                min_c = c
                min_i = i

        # Insert.
        ins = empty_i if empty_i >= 0 else min_i
        st.toks[base + ins] = tokp1
        st.cnts[base + ins] = 1

    def _roll(self, st: _NGramState, tok_id: int) -> None:
        """Update rolling N-gram fingerprint with the next token id."""
        tp1 = int(tok_id) + 1
        if st.filled < int(self.n):
            st.window[st.wpos] = tp1
            st.wpos = (st.wpos + 1) % int(self.n)
            st.filled += 1
            st.fp = (st.fp * int(self._base) + tp1) & int(self._mask64)
            return

        # Remove oldest contribution (rolling hash).
        old = int(st.window[st.wpos])
        st.window[st.wpos] = tp1
        st.wpos = (st.wpos + 1) % int(self.n)
        st.fp = (st.fp * int(self._base) + tp1 - (old * int(self._base_pow_n))) & int(self._mask64)

    def forward(self, logits: Tensor, *, ctx: object | None = None) -> Tensor:
        # logits: (B,T,V)
        if self.weight <= 0.0:
            return logits
        B, T, V = logits.shape
        if V != self.vocab_size:
            raise ValueError(f"Expected vocab_size={self.vocab_size}, got {V}")
        # Practical implementation targets batch_size=1.
        if B != 1:
            return logits

        st = self._ensure_state(ctx)
        if st is None:
            return logits

        # We require input token ids to update the rolling n-gram.
        input_ids = getattr(ctx, "input_ids", None) if ctx is not None else None
        if input_ids is None or not isinstance(input_ids, Tensor):
            return logits
        if input_ids.shape[0] != B or input_ids.shape[1] != T:
            # If shapes don't match, skip rather than silently corrupting state.
            return logits

        out = logits
        for t in range(int(T)):
            tok = int(input_ids[0, t].item())

            # Update rolling N-gram hash with current token.
            self._roll(st, tok_id=tok)

            # Read cache bias for predicting next token (only when window is full).
            if st.filled >= int(self.n):
                slot = int(st.fp % int(self.table_size))
                key = int(st.keys[slot])
                if key == ((int(st.fp) + 1) & int(self._mask64)):
                    # Compute sparse log-prob bias from counts.
                    items = list(self._iter_topm(st, slot))
                    if items:
                        total = float(sum(c for (_i, _tp1, c) in items))
                        if total > 0:
                            idxs: list[int] = []
                            vals: list[float] = []
                            for (_i, tp1, c) in items:
                                tok_id = int(tp1) - 1
                                if 0 <= tok_id < int(self.vocab_size) and c > 0:
                                    p = max(1e-9, float(c) / total)
                                    idxs.append(tok_id)
                                    vals.append(math.log(p) * float(self.weight))
                            if idxs:
                                out = out.clone()
                                row = out[0, t, :]
                                row.index_add_(
                                    0,
                                    torch.tensor(idxs, device=out.device, dtype=torch.long),
                                    torch.tensor(vals, device=out.device, dtype=out.dtype),
                                )

            # Write observation: current N-gram -> next token
            if t + 1 < int(T) and st.filled >= int(self.n):
                next_tok = int(input_ids[0, t + 1].item())
                slot = int(st.fp % int(self.table_size))
                want_key = ((int(st.fp) + 1) & int(self._mask64))
                if int(st.keys[slot]) != int(want_key):
                    # Evict slot on collision.
                    st.keys[slot] = int(want_key)
                    base = int(slot) * int(self.top_m)
                    for i in range(int(self.top_m)):
                        st.toks[base + i] = 0
                        st.cnts[base + i] = 0
                self._add_count(st, slot, tok_id=next_tok)

        _set_ngram_state(ctx, self._ctx_key, st)
        return out

