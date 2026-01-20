"""Cap'n Proto codec tests"""

import pytest
import torch

from core.event import EventEnvelope


def test_capnp_roundtrip():
    """Test encode/decode roundtrip for Cap'n Proto codec."""
    from core.event_codec import EventEncoder, EventDecoder

    encoder = EventEncoder()
    decoder = EventDecoder()

    # Create test event
    event = EventEnvelope(
        type="TestMessage",
        sender="test_agent",
        payload={"text": "Hello, Cap'n Proto!", "count": 42},
        priority=5,
        budget_ms=100,
        commitment_delta=1,
        commitment_id="test-commit-123",
    )

    # Encode
    encoded = encoder.encode(event)
    assert isinstance(encoded, torch.Tensor)
    assert encoded.dtype == torch.long
    assert encoded.ndim == 1
    assert encoded.numel() > 0

    # Decode
    decoded = decoder.decode(encoded)
    assert isinstance(decoded, EventEnvelope)

    # Verify fields
    assert decoded.type == event.type
    assert decoded.sender == event.sender
    assert decoded.payload == event.payload
    assert decoded.priority == event.priority
    assert decoded.budget_ms == event.budget_ms
    assert decoded.commitment_delta == event.commitment_delta
    assert decoded.commitment_id == event.commitment_id


def test_capnp_batch():
    """Test batch encode/decode for Cap'n Proto codec."""
    from core.event_codec import EventEncoder, EventDecoder

    encoder = EventEncoder()
    decoder = EventDecoder()

    events = [
        EventEnvelope(type="Msg1", sender="a", payload={"x": 1}),
        EventEnvelope(type="Msg2", sender="b", payload={"y": 2}),
        EventEnvelope(type="Msg3", sender="c", payload={"z": 3}),
    ]

    ids, mask = encoder.encode_padded(events)
    assert ids.shape[0] == 3
    assert mask.shape[0] == 3

    decoded = decoder.decode_padded(ids, mask)
    assert len(decoded) == 3
    for orig, dec in zip(events, decoded, strict=True):
        assert dec.type == orig.type
        assert dec.sender == orig.sender
        assert dec.payload == orig.payload
