"""Playwright utilities

This module provides Playwright browser automation utilities.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import BrowserContext, Page
from playwright.sync_api import sync_playwright
from playwright.sync_api import sync_playwright

from caramba.research.dba.colab_runner.base import ColabRunnerBase


class PlaywrightRunner(ColabRunnerBase):
    """Playwright runner."""
    def __init__(self) -> None:
        super().__init__()

        if not self.check_installed():
            self.install()

    def run(self) -> None:
        """Run the Colab runner."""
        pass

    def check_installed(self) -> bool:
        """Check if Playwright is installed."""
        try:
            sync_playwright()
            return True
        except ImportError:
            return False


    def install(self) -> None:
        """Install Playwright and browser."""
        print("Installing Playwright...")

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "playwright"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            if "externally-managed-environment" in result.stderr:
                print("\nUse: make colab-install")
                sys.exit(1)
            else:
                print(f"Failed: {result.stderr}")
                sys.exit(1)

        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        print("Playwright installed!")


    def get_profile_dir(self) -> Path:
        """Get the persistent Chrome profile directory."""
        return Path(tempfile.gettempdir()) / "colab_chrome_profile"


    def launch_browser(self, headless: bool = False) -> tuple["BrowserContext", "Page"]:
        """Launch browser with persistent profile.
        
        Returns:
            Tuple of (context, page)
        """        
        p = sync_playwright().start()
        profile_dir = self.get_profile_dir()

        try:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=headless,
                channel="chrome",
                viewport={"width": 1728, "height": 958},
            )
            page = context.pages[0] if context.pages else context.new_page()
            print("Using Chrome with persistent profile")
        except Exception as e:
            print(f"Chrome unavailable, using Chromium: {e}")
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(viewport={"width": 1728, "height": 958})
            page = context.new_page()

        return context, page
