"""Event codec package

Provides stable event encoders/decoders and stream parsers for event-native systems.
This package is the canonical home for codecs; `core/event_codec.py` re-exports
public symbols for backwards compatibility.
"""

from caramba.core.event_codec.json_codec import JsonEventDecoder, JsonEventEncoder
from caramba.core.event_codec.binary_codec import BinaryEventDecoder, BinaryEventEncoder
from caramba.core.event_codec.stream_parser import BinaryStreamParser

# Backwards-compatible aliases (v1 naming).
EventEncoder = JsonEventEncoder
EventDecoder = JsonEventDecoder

__all__ = [
    "BinaryEventDecoder",
    "BinaryEventEncoder",
    "BinaryStreamParser",
    "EventDecoder",
    "EventEncoder",
    "JsonEventDecoder",
    "JsonEventEncoder",
]

