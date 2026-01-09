"""Platform improve process

Implements an agentic development-team workflow that performs all implementation
inside an isolated Docker container (clone → branch → patch → test → PR). This
prevents accidental edits to the host repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.genai import types
from pydantic import BaseModel, Field

from caramba.ai.agent import Agent
from caramba.ai.process import Process
from caramba.ai.process.platform_improve.docker_workspace import DockerWorkspace
from caramba.config.platform_improve import PlatformImproveProcessConfig
from caramba.console import logger


class IdeaOutput(BaseModel):
    """Structured output for ideator agents."""

    title: str = Field(description="Short title for the improvement")
    rationale: str = Field(description="Why this improvement matters and its expected impact")
    risk: str = Field(description="Potential risks and how to mitigate them")
    verification: str = Field(description="How to verify the improvement works")
    files_to_touch: str = Field(description="Comma-separated list of files that will need to be modified")


class FileSelectionOutput(BaseModel):
    """Structured output for file selector agent."""

    files: list[str] = Field(description="List of relative file paths to inspect")


class VerdictOutput(BaseModel):
    """Structured output for verifier agent."""

    ok: bool = Field(description="True if the change passes verification, false otherwise")
    feedback: str = Field(description="Actionable feedback for the developer if ok is false")


@dataclass(frozen=True)
class VerificationReport:
    """Verification report

    Collects deterministic command outputs so the verifier agent can gate merges
    on real execution results.
    """

    outputs: list[dict[str, Any]]

    def ok(self) -> bool:
        """Success flag

        Returns True only when every command succeeded.
        """

        return all(bool(x.get("ok")) for x in self.outputs)


class PlatformImprove(Process):
    """Platform improve process

    Coordinates leader/ideators/developer/verifier and executes the implementation
    inside a Docker workspace.
    """

    def __init__(self, *, agents: dict[str, Agent], process: PlatformImproveProcessConfig) -> None:
        super().__init__(agents, name=process.name or "platform_improve")
        self.process = process

    async def run(self) -> None:
        """Run workflow

        Executes the full loop in the configured container workspace.
        """

        config = self.process
        team = self.resolveTeam()
        workspace = DockerWorkspace(
            image=config.workspace.image,
            workdir=config.workspace.workdir,
            name_prefix=config.workspace.container_name,
        )
        workspace.start()
        try:
            await self.runLoop(team=team, workspace=workspace)
        finally:
            workspace.close()

    async def runLoop(self, *, team: dict[str, Agent], workspace: DockerWorkspace) -> None:
        """Run loop

        Drives ideation → plan → implement → verify; repeats until success.
        """

        config = self.process
        workspace.ensureRepoCloned(
            repo_url=config.workspace.repo_url,
            repo_dir=config.workspace.repo_dir,
            base_branch=config.base_branch,
        )
        ideas = await self.collectIdeas(team=team)
        plan = await self.makePlan(team=team, ideas=ideas)

        for round_index in range(int(config.max_review_rounds) + 1):
            logger.header("Dev round", f"{round_index + 1}/{int(config.max_review_rounds) + 1}")
            branch = self.branchName(prefix=config.branch_prefix, round_index=round_index)
            workspace.createBranch(repo_dir=config.workspace.repo_dir, branch_name=branch)
            patch = await self.developerPatch(team=team, workspace=workspace, plan=plan)
            workspace.applyPatch(repo_dir=config.workspace.repo_dir, patch_text=patch)
            report = self.verify(workspace=workspace)
            verdict = await self.verdict(team=team, workspace=workspace, plan=plan, report=report)
            if report.ok() and verdict.ok:
                await self.openPrIfConfigured(workspace=workspace, plan=plan, branch=branch)
                return
            plan = self.feedbackPlan(plan=plan, report=report, verdict=verdict)

        raise RuntimeError("platform_improve did not converge within max_review_rounds")

    def resolveTeam(self) -> dict[str, Agent]:
        """Resolve team

        Ensures all required roles exist in the process team mapping.
        """

        config = self.process
        required = [config.leader, config.file_selector, config.developer, config.verifier]
        for role in required:
            if role not in self.agents:
                raise KeyError(f"Missing required team role: {role}")
        return self.agents

    async def collectIdeas(self, *, team: dict[str, Agent]) -> list[IdeaOutput]:
        """Collect ideas

        Requests one structured idea from each ideator. The ideator personas have
        output_schema set, so they return valid JSON matching IdeaOutput.
        """

        config = self.process
        ideators = [team[k] for k in config.ideators]
        if not ideators:
            raise ValueError("platform_improve requires at least one ideator")
        prompt = (
            f"Topic: {config.topic}\n\n"
            "Propose ONE high-impact improvement for the Caramba platform. "
            "Use your tools to explore the codebase before proposing."
        )
        ideas: list[IdeaOutput] = []
        for agent in ideators:
            text = await agent.run_async(types.Content(role="user", parts=[types.Part(text=prompt)]))
            idea = IdeaOutput.model_validate_json(text)
            ideas.append(idea)
        return ideas

    async def makePlan(self, *, team: dict[str, Agent], ideas: list[IdeaOutput]) -> str:
        """Make plan

        Leader selects exactly one improvement and produces a deterministic plan.
        """

        config = self.process
        leader = team[config.leader]
        ideas_json = [idea.model_dump() for idea in ideas]
        prompt = (
            "Pick EXACTLY ONE improvement and output a plan.\n"
            "Return plain text (no JSON), with acceptance criteria and verification commands.\n\n"
            f"Topic: {config.topic}\n\nIdeas:\n{json.dumps(ideas_json, ensure_ascii=False)}\n"
        )
        return await leader.run_async(types.Content(role="user", parts=[types.Part(text=prompt)]))

    async def developerPatch(self, *, team: dict[str, Agent], workspace: DockerWorkspace, plan: str) -> str:
        """Developer patch

        File selector chooses files to inspect, then developer receives their contents
        and returns a git diff.
        """

        config = self.process
        file_selector = team[config.file_selector]
        developer = team[config.developer]

        # File selector has output_schema set, returns valid JSON matching FileSelectionOutput
        file_prompt = f"Select the files that need to be inspected for this plan:\n\n{plan}"
        files_text = await file_selector.run_async(types.Content(role="user", parts=[types.Part(text=file_prompt)]))
        file_selection = FileSelectionOutput.model_validate_json(files_text)

        context = self.readFiles(workspace=workspace, files=file_selection.files)
        patch_prompt = (
            "Return ONLY a git diff patch starting with 'diff --git'. No prose.\n\n"
            f"PLAN:\n{plan}\n\nFILES:\n{context}\n"
        )
        patch_text = await developer.run_async(types.Content(role="user", parts=[types.Part(text=patch_prompt)]))
        return self.requirePatch(text=patch_text)

    def readFiles(self, *, workspace: DockerWorkspace, files: list[str]) -> str:
        """Read files

        Reads requested files from the container clone and formats them for the developer.
        """

        config = self.process
        repo_root = f"{config.workspace.workdir}/{config.workspace.repo_dir}"
        blocks: list[str] = []
        for rel in files:
            rel_path = str(rel).lstrip("./")
            path = f"{repo_root}/{rel_path}"
            text = workspace.readText(path=path, max_chars=int(config.max_chars_per_file))
            blocks.append(f"\n=== {rel_path} ===\n{text}\n")
        return "\n".join(blocks).strip()

    def verify(self, *, workspace: DockerWorkspace) -> VerificationReport:
        """Verify

        Runs configured verification commands inside the container clone.
        """

        config = self.process
        repo_root = f"{config.workspace.workdir}/{config.workspace.repo_dir}"
        outputs: list[dict[str, Any]] = []
        for cmd in config.tests:
            out = workspace.run(["bash", "-lc", cmd], cwd=repo_root)
            outputs.append(
                {
                    "cmd": cmd,
                    "ok": out.ok(),
                    "exit_code": out.exit_code,
                    "stdout": out.stdout,
                    "stderr": out.stderr,
                }
            )
        return VerificationReport(outputs=outputs)

    async def verdict(
        self, *, team: dict[str, Agent], workspace: DockerWorkspace, plan: str, report: VerificationReport
    ) -> VerdictOutput:
        """Verifier verdict

        Verifier agent gates on plan + diff + command outputs. The verifier persona
        has output_schema set, so it returns valid JSON matching VerdictOutput.
        """

        config = self.process
        verifier = team[config.verifier]
        diff_text = workspace.diff(repo_dir=config.workspace.repo_dir)
        prompt = (
            "Review this change. Hard rule: failing commands block approval.\n\n"
            f"PLAN:\n{plan}\n\nDIFF:\n{diff_text}\n\nCOMMANDS:\n{json.dumps(report.outputs, ensure_ascii=False)}\n"
        )
        text = await verifier.run_async(types.Content(role="user", parts=[types.Part(text=prompt)]))
        return VerdictOutput.model_validate_json(text)

    async def openPrIfConfigured(self, *, workspace: DockerWorkspace, plan: str, branch: str) -> None:
        """Open PR

        Pushes and creates PR using GitHub CLI inside the container when configured.
        """

        config = self.process
        if not bool(config.open_pr):
            return
        token = self.readToken(path=config.workspace.github_token_path)
        workspace.ghAuthWithToken(token_text=token)
        workspace.commitAll(repo_dir=config.workspace.repo_dir, message=f"{config.pr_title_prefix}{config.name}")
        workspace.pushBranch(repo_dir=config.workspace.repo_dir, remote=config.workspace.git_remote, branch_name=branch)
        title = f"{config.pr_title_prefix}{config.name}"
        body = f"## Summary\n\n{plan}\n"
        url = workspace.ghCreatePr(
            repo_dir=config.workspace.repo_dir,
            base_branch=config.base_branch,
            title=title,
            body=body,
        )
        logger.success(f"PR created: {url}")

    def readToken(self, *, path: str) -> str:
        """Read token

        Reads a GitHub token from a host file path specified in the manifest.
        """

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"GitHub token file not found: {p}. Provide it via process.workspace.github_token_path."
            )
        token = p.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"GitHub token file is empty: {p}")
        return token

    def branchName(self, *, prefix: str, round_index: int) -> str:
        """Branch name

        Names branches deterministically per round to avoid collisions.
        """

        safe_prefix = str(prefix).rstrip("/").replace(" ", "-")
        return f"{safe_prefix}/round-{round_index + 1}"

    def feedbackPlan(self, *, plan: str, report: VerificationReport, verdict: VerdictOutput) -> str:
        """Feedback plan

        Appends verifier feedback and command outputs to guide the next round.
        """

        return plan + "\n\nVERIFIER FEEDBACK:\n" + verdict.feedback + "\n\nCOMMAND OUTPUTS:\n" + json.dumps(
            report.outputs, ensure_ascii=False
        )

    def requirePatch(self, *, text: str) -> str:
        """Require patch

        Ensures the developer returns a git diff. The workflow depends on this
        exact format to apply changes inside the container.
        """

        if not text.lstrip().startswith("diff --git"):
            raise ValueError("Developer must return a git diff starting with 'diff --git'")
        return text.strip() + "\n"

