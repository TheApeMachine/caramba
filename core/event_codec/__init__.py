"""Event codec package

Provides stable event encoders/decoders using Cap'n Proto for zero-copy serialization.
This package is the canonical home for the event codec.
"""

from core.event_codec.capnp_codec import (
    CapnpEventDecoder,
    CapnpEventEncoder,
)

# Aliases for backward compatibility / simpler naming
EventEncoder = CapnpEventEncoder
EventDecoder = CapnpEventDecoder

__all__ = [
    "CapnpEventDecoder",
    "CapnpEventEncoder",
    "EventDecoder",
    "EventEncoder",
]
