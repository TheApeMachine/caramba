"""Pytest bootstrap for config tests.

Depending on how pytest discovers the project root, the repository root may not
end up on `sys.path` (e.g. when `rootdir` is inferred as `./config`). The code
and tests in this repo use top-level imports like `import config.*`, so we
ensure the repository root is importable.
"""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

