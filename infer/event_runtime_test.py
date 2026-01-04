from __future__ import annotations

import unittest

import torch
from torch import Tensor, nn

from caramba.core.event import EventEnvelope
from caramba.infer.context import InferContext
from caramba.infer.event_runtime import CommitmentModeB, EventResponder, StreamModelRunner


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

    def forward(self, input_ids: Tensor, ctx: InferContext) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("expected input_ids (B,T)")
        if input_ids.size(0) != 1:
            raise ValueError("dummy model expects batch_size=1")
        B, T = int(input_ids.size(0)), int(input_ids.size(1))
        V = int(self._vocab_size)
        logits = torch.full((B, T, V), -1e9, device=input_ids.device, dtype=torch.float32)

        for t in range(T):
            tok = int(input_ids[0, t].item())
            if not bool(self._started):
                if tok == 10:  # delimiter
                    self._started = True
                    self._emitted = 0
                next_idx = 0
            else:
                if tok != 10:
                    self._emitted = int(self._emitted) + 1
                next_idx = int(self._emitted)

            nxt = int(self._out[next_idx]) if 0 <= next_idx < len(self._out) else 0
            logits[0, t, nxt] = 0.0

        if ctx.mosaic_aux_out is not None:
            # Always predict "neutral" commitment (class=1 -> delta=0).
            aux = torch.zeros((1, T, 3), device=input_ids.device, dtype=torch.float32)
            aux[:, :, 1] = 1.0
            ctx.mosaic_aux_out["mosaic_commitment_logits"] = aux

        return logits


class EventRuntimeTest(unittest.TestCase):
    def test_event_responder_decodes_json_and_provides_aux(self) -> None:
        out_json = (
            b'{"id":"out","ts":0.0,"type":"Message","sender":"agent","priority":0,'
            b'"payload":{"text":"ok"},"commitment_delta":0}'
        )
        model = _DummyByteModel(out_bytes=list(out_json))
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

