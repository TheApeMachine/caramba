"""Event codec package

Provides stable event encoders/decoders using Cap'n Proto for zero-copy serialization.
This package is the canonical home for the event codec.
"""

from caramba.core.event_codec.capnp_codec import (
    CapnpEventDecoder,
    CapnpEventEncoder,
    is_capnp_available,
)

# Aliases for backward compatibility / simpler naming
EventEncoder = CapnpEventEncoder
EventDecoder = CapnpEventDecoder

__all__ = [
    "CapnpEventDecoder",
    "CapnpEventEncoder",
    "EventDecoder",
    "EventEncoder",
    "is_capnp_available",
]
