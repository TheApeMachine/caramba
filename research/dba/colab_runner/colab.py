"""Colab runner

This module provides the main Colab runner that coordinates other runners.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from playwright.sync_api import Page

from research.dba.colab_runner.javascript import JavaScriptRunner
from research.dba.colab_runner.notebook import NotebookRunner


class ColabRunner:
    """Colab runner - coordinates JavaScriptRunner and NotebookRunner."""

    def __init__(self, page: Page, folder_id: str | None = None) -> None:
        self.page = page
        self.folder_id = folder_id
        self.javascript = JavaScriptRunner(page)
        self.notebook = NotebookRunner(page)

    def open_notebook(self) -> None:
        """Open Colab with a new notebook."""
        if self.folder_id:
            url = f"https://colab.research.google.com/#create=true&folderId={self.folder_id}"
        else:
            url = "https://colab.research.google.com/#create=true"
        self.page.goto(url, wait_until="networkidle")

    def inject_script(self) -> None:
        """Inject the caramba.js script into the page."""
        self.javascript.run()

    def add_cell(self, cell: dict) -> None:
        """Add a cell to the notebook."""
        self.notebook.add_cell(cell)

    def run_cell(self, cell_id: str) -> None:
        """Run a cell in the notebook."""
        self.notebook.run_cell(cell_id)

    def get_cell_ids(self) -> list[str]:
        """Get all cell IDs from the notebook."""
        return self.javascript.get_cell_ids()

    def wait_for_cell(self, cell_id: str, timeout: int = 600) -> bool:
        """Wait for a cell to complete execution."""
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(2)
            status = self.javascript.get_cell_status(cell_id)
            if status == "idle":
                return True
            if self.has_error():
                return False
        return False

    def has_error(self) -> bool:
        """Check if the page contains an error."""
        content = self.page.content()
        return "Traceback" in content and "Error" in content

    def is_complete(self) -> bool:
        """Check if benchmark is complete."""
        content = self.page.content()
        return "Benchmark complete!" in content

    def screenshot(self, name: str) -> Path:
        """Take a screenshot."""
        path = Path(tempfile.gettempdir()) / f"{name}.png"
        self.page.screenshot(path=str(path))
        return path

    def run(self) -> None:
        """Run the Colab runner (placeholder for full automation)."""
        pass
