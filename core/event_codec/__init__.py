"""Event codec package

Provides stable event encoders/decoders and stream parsers for event-native systems.
This package is the canonical home for codecs.
"""

from caramba.core.event_codec.json_codec import JsonEventDecoder, JsonEventEncoder
from caramba.core.event_codec.binary_codec import BinaryEventDecoder, BinaryEventEncoder
from caramba.core.event_codec.stream_parser import BinaryStreamParser

__all__ = [
    "BinaryEventDecoder",
    "BinaryEventEncoder",
    "BinaryStreamParser",
    "JsonEventDecoder",
    "JsonEventEncoder",
]

