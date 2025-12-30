"""Convenience entrypoint + compatibility package shim.

This repository historically used a "flat" import style where modules like
`cli.py`, `layer/`, `trainer/`, etc. live at the repo root.

When your working directory is the repo root, `python -m caramba` expects a
`caramba` module on `sys.path`. Providing this file makes that command work
without requiring installation.

Note: When running from the parent directory (where the `caramba/` folder is on
`sys.path`), `python -m caramba` will execute `caramba/__main__.py` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from cli import main

# Compatibility: some code paths import `caramba.<submodule>` (package-style).
# When executed from the repo root, Python will import this file as the `caramba`
# module, which is *not* a package by default. Setting `__path__` makes it behave
# like a package rooted at this directory, so `import caramba.trainer` works.
#
# This is safe because our real "package" layout is effectively the repo root.
__path__ = [str(Path(__file__).resolve().parent)]  # type: ignore[name-defined]
if __path__[0] not in sys.path:
    sys.path.insert(0, __path__[0])


if __name__ == "__main__":
    raise SystemExit(main())

