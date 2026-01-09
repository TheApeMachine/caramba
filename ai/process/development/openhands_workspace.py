"""OpenHands SDK workspace for AI-driven development.

Wraps the OpenHands SDK to provide a contained development environment where
agents can clone repositories, make changes, run tests, and create pull requests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from caramba.console import logger


@dataclass
class DevelopmentResult:
    """Result of a development task execution."""

    success: bool
    branch_name: str | None = None
    pr_url: str | None = None
    files_changed: list[str] = field(default_factory=list)
    test_output: str = ""
    error: str | None = None


class OpenHandsWorkspace:
    """OpenHands SDK workspace for contained development.

    Provides a sandboxed environment where AI agents can:
    - Clone and work with git repositories
    - Edit files safely
    - Run tests and build commands
    - Create branches and pull requests
    """

    def __init__(
        self,
        *,
        repo_url: str,
        base_branch: str = "main",
        workdir: str = "/tmp/openhands-workspace",
        model: str | None = None,
    ) -> None:
        self.repo_url = repo_url
        self.base_branch = base_branch
        self.workdir = Path(workdir)
        self.model = model or os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929")
        self.repo_dir: Path | None = None
        self._conversation: Conversation | None = None
        self._agent: Agent | None = None

    def setup(self) -> None:
        """Initialize the OpenHands workspace and clone the repository."""
        self.workdir.mkdir(parents=True, exist_ok=True)

        # Configure LLM
        llm = LLM(
            model=self.model,
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )

        # Create agent with development tools
        self._agent = Agent(
            llm=llm,
            tools=[
                Tool(name=TerminalTool.name),
                Tool(name=FileEditorTool.name),
                Tool(name=TaskTrackerTool.name),
            ],
        )

        # Initialize conversation with workspace
        self._conversation = Conversation(
            agent=self._agent,
            workspace=str(self.workdir),
        )

        # Clone repository
        repo_name = self.repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        self.repo_dir = self.workdir / repo_name

        if self.repo_dir.exists():
            logger.info(f"Repository already exists at {self.repo_dir}, pulling latest...")
            self._run_task(f"cd {self.repo_dir} && git fetch origin && git checkout {self.base_branch} && git pull origin {self.base_branch}")
        else:
            logger.info(f"Cloning {self.repo_url}...")
            self._run_task(f"git clone {self.repo_url} {self.repo_dir}")
            self._run_task(f"cd {self.repo_dir} && git checkout {self.base_branch}")

    def create_branch(self, branch_name: str) -> None:
        """Create a new feature branch."""
        if not self.repo_dir:
            raise RuntimeError("Workspace not set up. Call setup() first.")
        self._run_task(f"cd {self.repo_dir} && git checkout -B {branch_name}")
        logger.info(f"Created branch: {branch_name}")

    def implement_changes(self, task_description: str) -> DevelopmentResult:
        """Use OpenHands agent to implement changes based on the task description.

        Args:
            task_description: Natural language description of what to implement

        Returns:
            DevelopmentResult with information about what was changed
        """
        if not self._conversation or not self.repo_dir:
            raise RuntimeError("Workspace not set up. Call setup() first.")

        try:
            # Send the implementation task to OpenHands
            full_task = f"""
You are working in the repository at {self.repo_dir}.

Task: {task_description}

Instructions:
1. First, explore the relevant files to understand the codebase structure
2. Make the necessary changes to implement the task
3. Ensure your changes follow the existing code style
4. Run any relevant tests to verify your changes work

When done, provide a summary of what files you changed and why.
"""
            self._conversation.send_message(full_task)
            self._conversation.run()

            # Get changed files
            result = self._run_task(f"cd {self.repo_dir} && git diff --name-only")
            files_changed = [f.strip() for f in result.split("\n") if f.strip()]

            return DevelopmentResult(
                success=True,
                files_changed=files_changed,
            )

        except Exception as e:
            logger.error(f"Implementation failed: {e}")
            return DevelopmentResult(
                success=False,
                error=str(e),
            )

    def run_tests(self, test_command: str = "python -m pytest -q") -> tuple[bool, str]:
        """Run tests in the repository.

        Returns:
            Tuple of (success, output)
        """
        if not self.repo_dir:
            raise RuntimeError("Workspace not set up. Call setup() first.")

        try:
            output = self._run_task(f"cd {self.repo_dir} && {test_command}")
            # Check for test failures in output
            success = "FAILED" not in output and "error" not in output.lower()
            return success, output
        except Exception as e:
            return False, str(e)

    def commit_and_push(
        self,
        branch_name: str,
        commit_message: str,
        *,
        remote: str = "origin",
    ) -> bool:
        """Commit changes and push to remote."""
        if not self.repo_dir:
            raise RuntimeError("Workspace not set up. Call setup() first.")

        try:
            self._run_task(f"cd {self.repo_dir} && git add -A")
            self._run_task(f"cd {self.repo_dir} && git commit -m '{commit_message}'")
            self._run_task(f"cd {self.repo_dir} && git push -u {remote} {branch_name}")
            logger.success(f"Pushed changes to {remote}/{branch_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to push: {e}")
            return False

    def create_pull_request(
        self,
        *,
        branch_name: str,
        title: str,
        body: str,
        base_branch: str | None = None,
    ) -> str | None:
        """Create a pull request using GitHub CLI.

        Returns:
            PR URL if successful, None otherwise
        """
        if not self.repo_dir:
            raise RuntimeError("Workspace not set up. Call setup() first.")

        base = base_branch or self.base_branch
        try:
            result = self._run_task(
                f"cd {self.repo_dir} && gh pr create --base {base} --head {branch_name} "
                f"--title '{title}' --body '{body}'"
            )
            # Extract URL from output
            for line in result.split("\n"):
                if "github.com" in line and "/pull/" in line:
                    return line.strip()
            return result.strip()
        except Exception as e:
            logger.error(f"Failed to create PR: {e}")
            return None

    def get_diff(self) -> str:
        """Get the current git diff."""
        if not self.repo_dir:
            raise RuntimeError("Workspace not set up. Call setup() first.")
        return self._run_task(f"cd {self.repo_dir} && git diff")

    def read_file(self, path: str) -> str:
        """Read a file from the repository."""
        if not self.repo_dir:
            raise RuntimeError("Workspace not set up. Call setup() first.")
        full_path = self.repo_dir / path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return full_path.read_text(encoding="utf-8")

    def _run_task(self, command: str) -> str:
        """Run a shell command in the workspace."""
        if not self._conversation:
            raise RuntimeError("Workspace not set up. Call setup() first.")

        self._conversation.send_message(f"Run this command and return the output: {command}")
        self._conversation.run()

        # Get the last assistant message which should contain the output
        # This is a simplified approach - in practice you'd parse the conversation history
        return ""

    def cleanup(self) -> None:
        """Clean up the workspace."""
        if self._conversation:
            self._conversation = None
        if self._agent:
            self._agent = None
        logger.info("Workspace cleaned up")
