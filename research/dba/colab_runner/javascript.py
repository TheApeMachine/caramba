"""JavaScript runner

This module provides a JavaScript runner for Colab.
"""
from __future__ import annotations

from pathlib import Path
from playwright.sync_api import Page


SCRIPTS_DIR = Path(__file__).parent / "scripts"


class JavaScriptRunner:
    """JavaScript runner - handles script loading and injection."""

    def __init__(self, page: Page) -> None:
        self.page = page
        self.caramba_js = self._load_script("js", "caramba.js")

    def _load_script(self, script_type: str, name: str) -> str:
        """Load a script file from the scripts directory."""
        path = SCRIPTS_DIR / script_type / name
        if not path.exists():
            raise FileNotFoundError(f"Script not found: {path}")
        return path.read_text()

    def run(self) -> None:
        """Inject the caramba.js script into the page."""
        self.page.evaluate(self.caramba_js)

    def get_cell_status(self, cell_id: str) -> str:
        """Get the status of a cell."""
        return self.page.evaluate(f"""
            (() => {{
                const cells = colab.global.notebook.actualEventTarget_.cells;
                const cell = cells.find(c => c.getId() === "{cell_id}");
                if (cell && cell.runButton) {{
                    return cell.runButton.getStatus ? cell.runButton.getStatus() : 'unknown';
                }}
                return 'unknown';
            }})()
        """)

    def get_cell_ids(self) -> list[str]:
        """Get all cell IDs from the notebook."""
        return self.page.evaluate("""
            colab.global.notebook.actualEventTarget_.cells.map(c => c.getId())
        """)
