"""Benchmark runner.

This module provides the main benchmark runner that orchestrates
the Colab automation for running DBA benchmarks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from caramba.research.dba.colab_runner.base import ColabRunnerBase
from caramba.research.dba.colab_runner.cells import get_benchmark_cells
from caramba.research.dba.colab_runner.colab import ColabRunner
from caramba.research.dba.colab_runner.paths import normalize_drive_path
from caramba.research.dba.colab_runner.browser import PlaywrightRunner


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    checkpoint_dir: str
    results_dir: str = "/DBA/results"
    tests_per_category: int = 30
    seed: int = 42
    webhook: str | None = None
    github_repo: str = "theapemachine/caramba"
    github_branch: str = "main"
    folder_id: str | None = None
    headless: bool = False
    timeout: int = 3600

    def __post_init__(self) -> None:
        """Normalize paths after initialization."""
        self.checkpoint_dir = normalize_drive_path(self.checkpoint_dir)
        self.results_dir = normalize_drive_path(self.results_dir)

    def print_config(self) -> None:
        """Print configuration summary."""
        print("="*60)
        print("DBA Colab Benchmark Runner")
        print("="*60)
        print(f"\nConfig:")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Results: {self.results_dir}")
        print(f"  Tests/category: {self.tests_per_category}")
        if self.folder_id:
            print(f"  Folder ID: {self.folder_id}")


class BenchmarkRunner(ColabRunnerBase):
    """Benchmark runner that orchestrates Colab automation."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__()
        self.config = config
        self.playwright = PlaywrightRunner()
        self.cells = self._load_cells()

    def _load_cells(self) -> list[dict]:
        """Load benchmark cells from external files."""
        cells = get_benchmark_cells(
            checkpoint_dir=self.config.checkpoint_dir,
            results_dir=self.config.results_dir,
            tests_per_category=self.config.tests_per_category,
            seed=self.config.seed,
            webhook=self.config.webhook,
            github_repo=self.config.github_repo,
            github_branch=self.config.github_branch,
        )
        print(f"Loaded {len(cells)} cells")
        return cells

    def run(self) -> None:
        """Run the benchmark."""
        self.config.print_config()

        print(f"\nStarting Colab automation...")
        print(f"Cells: {len(self.cells)}, Timeout: {self.config.timeout}s")

        context, page = self.playwright.launch_browser(headless=self.config.headless)
        colab = ColabRunner(page, folder_id=self.config.folder_id)

        try:
            self._execute_benchmark(colab)

            if not self.config.headless:
                self._wait_for_user(colab)

        except Exception as e:
            print(f"\nError: {e}")
            colab.screenshot("colab_error")
            raise

        finally:
            context.close()

    def _execute_benchmark(self, colab: ColabRunner) -> None:
        """Execute the benchmark cells."""
        print("\n1. Opening Colab...")
        colab.open_notebook()

        print("\n2. Injecting script...")
        colab.inject_script()

        print("\n3. Adding cells...")
        for cell in self.cells:
            colab.add_cell(cell)
            print(f"   Added: {cell['description']}")

        print("\n4. Running cells...")
        self._run_all_cells(colab)

        screenshot = colab.screenshot("colab_final")
        print(f"\nScreenshot: {screenshot}")

    def _run_all_cells(self, colab: ColabRunner) -> None:
        """Run all cells sequentially."""
        start_time = time.time()
        cell_ids = colab.get_cell_ids()

        for i, cell_id in enumerate(cell_ids):
            if time.time() - start_time > self.config.timeout:
                print(f"\nTimeout after {self.config.timeout}s")
                break

            desc = self.cells[i]["description"] if i < len(self.cells) else f"Cell {i+1}"
            print(f"   [{i+1}/{len(self.cells)}] Running: {desc}")

            colab.run_cell(cell_id)
            success = colab.wait_for_cell(cell_id)

            elapsed = int(time.time() - start_time)
            print(f"      Done ({elapsed}s total)")

            if not success:
                print("   Error detected!")
                break

            if colab.is_complete():
                print("\n" + "="*60)
                print("BENCHMARK COMPLETE!")
                print("="*60)
                break

    def _wait_for_user(self, colab: ColabRunner) -> None:
        """Wait for user to close browser."""
        print("\nBrowser open. Ctrl+C to close.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
