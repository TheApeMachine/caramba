"""Path utilities

This module provides path normalization utilities for Google Drive.
"""
from __future__ import annotations


DRIVE_PREFIX = "/content/drive/MyDrive"


def normalize_drive_path(path: str) -> str:
    """Normalize a path to include the Google Drive prefix."""
    if not path.startswith(DRIVE_PREFIX):
        return f"{DRIVE_PREFIX}{path}"
    return path
