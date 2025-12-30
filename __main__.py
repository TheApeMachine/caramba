"""Console-script entrypoint for the caramba package.

This allows running caramba as `python -m caramba` or via the console script.
All commands are routed through the unified CLI.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Support running as an installed package (`python -m caramba`) while keeping the
# codebase's "flat" import style (e.g. `from cli import ...`, `from layer import ...`).
# When executed as a package, those modules live under this directory, so we
# ensure the package directory itself is on sys.path.
_PKG_DIR = str(Path(__file__).resolve().parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from cli import main as cli_main


def main(argv: list[str] | None = None) -> None:
    """Entrypoint for the `caramba` console script.

    Routes everything through the unified CLI to ensure consistent behavior
    for the single manifest-driven entrypoint.
    """
    exit_code = cli_main(argv)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
