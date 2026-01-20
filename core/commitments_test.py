from __future__ import annotations

import unittest

from core.commitments import CommitmentLedger
from core.event import EventEnvelope


class CommitmentLedgerTest(unittest.TestCase):
    def test_open_autogenerates_id_when_missing(self) -> None:
        led = CommitmentLedger()
        ev = EventEnvelope(
            type="Message",
            payload={"text": "I will do it"},
            sender="agent",
            commitment_delta=+1,
            commitment_id=None,
            id="e1",
            ts=1.0,
        )
        out = led.apply(ev)
        self.assertIsNotNone(out.commitment_id)
        self.assertEqual(led.metrics.opens, 1)
        self.assertEqual(len(led.open_ids()), 1)

    def test_close_without_id_closes_most_recent_for_sender(self) -> None:
        led = CommitmentLedger()
        open1 = EventEnvelope(
            type="Message",
            payload={"text": "open1"},
            sender="agent",
            commitment_delta=+1,
            commitment_id="c1",
            id="e1",
            ts=1.0,
        )
        open2 = EventEnvelope(
            type="Message",
            payload={"text": "open2"},
            sender="agent",
            commitment_delta=+1,
            commitment_id="c2",
            id="e2",
            ts=2.0,
        )
        led.apply(open1)
        led.apply(open2)

        close = EventEnvelope(
            type="Message",
            payload={"text": "done"},
            sender="agent",
            commitment_delta=-1,
            commitment_id=None,
            id="e3",
            ts=3.0,
        )
        out = led.apply(close)
        self.assertEqual(out.commitment_id, "c2")
        self.assertEqual(led.metrics.closes, 1)
        self.assertEqual(led.metrics.close_without_id, 1)
        self.assertEqual(led.open_ids(), ["c1"])

    def test_idle_increments_metric_when_open_commitment_exists(self) -> None:
        led = CommitmentLedger()
        led.apply(
            EventEnvelope(
                type="Message",
                payload={"text": "open"},
                sender="agent",
                commitment_delta=+1,
                commitment_id="c1",
                id="e1",
                ts=1.0,
            )
        )
        led.apply(
            EventEnvelope(
                type="Idle",
                payload={},
                sender="agent",
                commitment_delta=0,
                id="e2",
                ts=2.0,
            )
        )
        self.assertEqual(led.metrics.idle_with_open_commitments, 1)

    def test_close_without_open_raises(self) -> None:
        led = CommitmentLedger()
        with self.assertRaises(ValueError):
            led.apply(
                EventEnvelope(
                    type="Message",
                    payload={"text": "close"},
                    sender="agent",
                    commitment_delta=-1,
                    commitment_id=None,
                    id="e1",
                    ts=1.0,
                )
            )

