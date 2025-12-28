"""Pytest configuration.

Caramba's codebase intentionally uses top-level module imports (e.g. `model`,
`layer`, `benchmark`) to keep runtime imports simple when executing as
`python -m caramba`.

When pytest runs from the package parent, the repository root may not be on
`sys.path`, which breaks those top-level imports. This conftest ensures the
repo root is importable for tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

