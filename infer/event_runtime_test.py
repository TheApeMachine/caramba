from __future__ import annotations

import unittest

import torch
from torch import Tensor, nn

from core.event import EventEnvelope
from core.event_codec import EventEncoder
from infer.context import InferContext
from infer.event_runtime import CommitmentModeB, EventResponder, StreamModelRunner


class _DummyByteModel(nn.Module):
    """A tiny deterministic byte-level next-token model for runtime tests.

    It starts emitting a fixed JSON byte sequence right after seeing the prompt
    delimiter byte (newline). This exercises EventResponder's buffering logic.
    """

    def __init__(self, *, out_bytes: list[int], vocab_size: int = 256) -> None:
        super().__init__()
        self._out = [int(b) for b in out_bytes]
        self._vocab_size = int(vocab_size)
        self._started = False
        self._emitted = 0

    def _compute_idx(self, t: int, T: int, ctx: InferContext) -> int:
        """Compute output-byte index for prompt vs generation phases."""
        if T > 1:
            # Prompt phase: remember prompt length and only emit at the very end.
            self._prompt_len = T
            return 0 if t == T - 1 else -1

        # Generation phase: ctx.pos_offset points at the current absolute position.
        if not hasattr(self, "_prompt_len"):
            self._prompt_len = 0
        return int(ctx.pos_offset) - int(self._prompt_len) + 1

    def forward(self, input_ids: Tensor, ctx: InferContext) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("expected input_ids (B,T)")
        if input_ids.size(0) != 1:
            raise ValueError("dummy model expects batch_size=1")
        B, T = int(input_ids.size(0)), int(input_ids.size(1))
        V = int(self._vocab_size)
        logits = torch.full((B, T, V), -1e9, device=input_ids.device, dtype=torch.float32)

        # In the first forward (prompt), we don't care about logits much.
        # But once we start generating (tokens being appended to the prompt),
        # we want the model to emit out_bytes.
        # Responder's first generate call happens after prompt of length P.
        # The first generated token should be out_bytes[0].
        # In that call, ctx.pos_offset = P, input_ids = [[last_token_of_prompt]].
        # Actually, responder does:
        # 1. forward(prompt) -> get next_logits
        # 2. tok = sample(next_logits)
        # 3. while ... forward(tok) -> next_logits

        # So when ctx.pos_offset is 0, we want to predict out_bytes[0] at the end of the prompt.
        # The prompt length is P. The last logit of the prompt (at T-1) should predict out_bytes[0].
        for t in range(T):
            # idx is what this position should predict.
            # position P in the sequence should predict out_bytes[0] if prompt length is P.
            # position is ctx.pos_offset + t.
            # We don't know the exact prompt length yet, but we know when we are generating.
            # Generating happens when T=1 and pos > 0.
            idx = self._compute_idx(t, T, ctx)

            nxt = int(self._out[idx]) if 0 <= idx < len(self._out) else 0
            logits[0, t, nxt] = 0.0

        if ctx.memblock_aux_out is not None:
            # Always predict "neutral" commitment (class=1 -> delta=0).
            aux = torch.zeros((1, T, 3), device=input_ids.device, dtype=torch.float32)
            aux[:, :, 1] = 1.0
            ctx.memblock_aux_out["mosaic_commitment_logits"] = aux

        return logits


class EventRuntimeTest(unittest.TestCase):
    def test_event_responder_decodes_event_and_provides_aux(self) -> None:
        out_event = EventEnvelope(
            type="Message", payload={"text": "ok"}, sender="agent", id="out", ts=0.0
        )
        encoder = EventEncoder()
        out_bytes = encoder.encode(out_event).tolist()

        model = _DummyByteModel(out_bytes=list(out_bytes))
        runner = StreamModelRunner(model=model, ctx=InferContext(caches=[]), collect_aux=True)
        responder = EventResponder(runner=runner, max_new_tokens=2048)

        inbound = EventEnvelope(type="Message", payload={"text": "hi"}, sender="user", id="in", ts=0.0)
        ev, aux = responder.respond(inbound)

        self.assertEqual(ev.type, "Message")
        self.assertEqual(ev.sender, "agent")
        self.assertEqual(ev.payload, {"text": "ok"})
        self.assertIsNotNone(aux)
        assert aux is not None
        self.assertIn("mosaic_commitment_logits", aux)

        injected = CommitmentModeB().inject(ev, aux=aux)
        self.assertEqual(injected.commitment_delta, 0)

