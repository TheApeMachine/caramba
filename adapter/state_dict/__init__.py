"""State-dict adapters.

Apply external checkpoint state_dict mappings to manifest-built models by
composing a schema (data) and policy objects (behavior).
"""

from __future__ import annotations

from caramba.adapter.state_dict.transformer import AdapterStateDictTransformer

__all__ = ["AdapterStateDictTransformer"]
