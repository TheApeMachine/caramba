from __future__ import annotations

import unittest

import torch

from caramba.core.event_codec.binary_codec import BinaryEventDecoder, BinaryEventEncoder, BinaryFrame
from caramba.core.event_codec.stream_parser import BinaryStreamParser


class BinaryCodecTest(unittest.TestCase):
    def test_round_trip_frame(self) -> None:
        enc = BinaryEventEncoder()
        dec = BinaryEventDecoder()
        frame = BinaryFrame(type_id=7, payload=b"abc")
        ids = enc.encode_frame(frame)
        out = dec.decode_frame(ids)
        self.assertEqual(out.type_id, 7)
        self.assertEqual(out.payload, b"abc")

    def test_stream_parser(self) -> None:
        enc = BinaryEventEncoder()
        parser = BinaryStreamParser()
        ids = enc.encode_frame(BinaryFrame(type_id=1, payload=b"hello")).to(dtype=torch.uint8).cpu()
        raw = bytes(int(x) for x in ids.tolist())

        a = parser.feed(raw[:3])
        self.assertEqual(a, [])
        b = parser.feed(raw[3:])
        self.assertEqual(len(b), 1)
        self.assertEqual(b[0].type_id, 1)
        self.assertEqual(b[0].payload, b"hello")

