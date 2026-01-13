"""Event codec package

Provides stable event encoders/decoders using Cap'n Proto.

Note: "zero-copy" is achievable for *buffer views* in some paths (e.g. decoding
from a `uint8` buffer without re-materializing bytes). If you request `torch.long`
token ids, dtype conversion necessarily allocates/copies.
This package is the canonical home for the event codec.
"""

from caramba.core.event_codec.capnp_codec import (
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
