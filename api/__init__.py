"""Caramba API package.

This package exposes the FastAPI application used by the CLI `serve` command.
"""

from __future__ import annotations

from api.app import app

__all__ = ["app"]

