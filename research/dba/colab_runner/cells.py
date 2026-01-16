"""Cell loading and management

This module provides cell loading from external Python files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


SCRIPTS_DIR = Path(__file__).parent / "scripts"

CELL_DESCRIPTIONS = [
    "Configuration",
    "Check GPU and Mount Drive",
    "Install Dependencies",
    "Clone Repository",
    "Discover Checkpoints",
    "Run Benchmark",
    "Send Notification",
    "Display Results",
]


def load_script(script_type: str, name: str) -> str:
    """Load a script file from the scripts directory."""
    path = SCRIPTS_DIR / script_type / name
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")
    return path.read_text()


def load_cell(index: int, **kwargs: Any) -> dict:
    """Load a Python cell script and apply variable substitution."""
    code = load_script("python", f"cell{index}.py")
    
    # Simple template substitution for configuration values
    for key, value in kwargs.items():
        code = code.replace(f"{{{key}}}", str(value))
    
    return {
        "description": CELL_DESCRIPTIONS[index] if index < len(CELL_DESCRIPTIONS) else f"Cell {index}",
        "code": code,
    }


def get_benchmark_cells(
    checkpoint_dir: str,
    results_dir: str,
    tests_per_category: int = 30,
    seed: int = 42,
    webhook: str | None = None,
    github_repo: str = "theapemachine/caramba",
    github_branch: str = "main",
) -> list[dict]:
    """Load code cells for the DBA benchmark notebook from external files.
    
    Returns:
        List of dicts with 'code' and 'description' keys.
    """
    template_vars = {
        "checkpoint_dir": checkpoint_dir,
        "results_dir": results_dir,
        "tests_per_category": tests_per_category,
        "seed": seed,
        "webhook": webhook or "",
        "github_repo": github_repo,
        "github_branch": github_branch,
    }
    
    cells = []
    python_dir = SCRIPTS_DIR / "python"
    cell_files = sorted(python_dir.glob("cell*.py"), key=lambda p: int(p.stem.replace("cell", "")))
    
    for i, _ in enumerate(cell_files):
        cells.append(load_cell(i, **template_vars))
    
    return cells
