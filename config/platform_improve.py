"""Platform improve configuration

Defines the manifest-driven configuration for the platform_improve process,
including a Docker workspace that clones the repo and performs all mutations
inside the container.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DockerWorkspaceConfig(BaseModel):
    """Docker workspace configuration

    Specifies how the agentic dev team should create an isolated container
    workspace for cloning, building, testing, and opening PRs.
    """

    image: str = Field(default="caramba-dev-agent:latest", description="Docker image name for dev workspace")
    container_name: str = Field(default="caramba-dev-agent", description="Container name (must be unique)")
    workdir: str = Field(default="/work", description="Working directory inside container")
    repo_url: str = Field(description="Git clone URL (HTTPS or SSH)")
    repo_dir: str = Field(default="repo", description="Directory name for cloned repo inside workdir")
    git_remote: str = Field(default="origin", description="Git remote name used for pushing")
    github_token_path: str = Field(default=".secrets/github_token.txt", description="Host path to GitHub token file")


class PlatformImproveProcessConfig(BaseModel):
    """Platform improve process configuration

    Declares the agentic development workflow and verification gates. All
    implementation work happens in the configured Docker workspace.
    """

    type: Literal["platform_improve"] = "platform_improve"
    name: str | None = None
    topic: str = "Propose and implement one high-impact improvement to the Caramba platform."

    ingest_repo: bool = True
    ingest_models: bool = True
    index_namespace: str = "main"
    ingest_agent: str = "research_team_leader"
    max_files: int = Field(default=250, ge=1)
    max_chars_per_file: int = Field(default=4000, ge=256)

    leader: str = "research_team_leader"
    ideators: list[str] = Field(default_factory=list)
    file_selector: str = "file_selector"
    developer: str = "developer"
    verifier: str = "verifier"

    base_branch: str = "main"
    branch_prefix: str = "agent/platform-improve"
    tests: list[str] = Field(default_factory=lambda: ["python -m pytest -q"])
    max_review_rounds: int = Field(default=2, ge=0)
    open_pr: bool = True
    pr_title_prefix: str = "[caramba] "

    workspace: DockerWorkspaceConfig

    @model_validator(mode="after")
    def defaultName(self) -> "PlatformImproveProcessConfig":
        if not self.name:
            self.name = self.type
        if not self.ideators:
            raise ValueError("platform_improve requires ideators (non-empty list)")
        return self

