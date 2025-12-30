"""Agent process configuration.

This extends the manifest-driven approach to the built-in agent system:
- define a *team* (keys -> persona yaml names)
- define one or more *processes* (discussion, paper workflows, etc.)
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, RootModel


class AgentTeamConfig(RootModel[dict[str, str]]):
    """Mapping of team role keys to persona names.

    Example:
        team:
          research_team_leader: research_lead
          developer: developer
    """


class DiscussionProcessConfig(BaseModel):
    """Configuration for the `discussion` agent process."""

    type: Literal["discussion"] = "discussion"
    name: str
    leader: str
    topic: str
    prompts_dir: str = "config/prompts"
    max_rounds: int = Field(default=12, ge=1)


class PaperWriteProcessConfig(BaseModel):
    """Draft/update a paper artifact using a writer persona."""

    type: Literal["paper_write"] = "paper_write"
    name: str
    writer: str
    # Optional: high-level goal/instructions for the writer.
    goal: str = ""
    # Where to read/write paper artifacts (usually under artifacts/).
    output_dir: str = "paper"


class PaperReviewProcessConfig(BaseModel):
    """Review a paper artifact using a reviewer persona."""

    type: Literal["paper_review"] = "paper_review"
    name: str
    reviewer: str
    strictness: str = "conference"
    max_proposed_experiments: int = Field(default=3, ge=0)
    # Optional reviewer-specific instructions to append.
    goal: str = ""


class ResearchLoopProcessConfig(BaseModel):
    """Orchestrate write → review → structural audit → (optionally) experiments."""

    type: Literal["research_loop"] = "research_loop"
    name: str
    leader: str
    writer: str
    reviewer: str
    max_iterations: int = Field(default=5, ge=1)
    # If true, allow the loop to run experiments automatically (Phase 2 readiness applies).
    auto_run_experiments: bool = False
    output_dir: str = "paper"


class CodeGraphSyncProcessConfig(BaseModel):
    """Ingest manifest/model topology into Graphiti graph memory."""

    type: Literal["code_graph_sync"] = "code_graph_sync"
    name: str
    agent: str
    # Prefix/namespace to avoid collisions across manifests/runs.
    index_namespace: str = "main"


class PlatformImproveProcessConfig(BaseModel):
    """End-to-end platform improvement pipeline.

    Orchestrates:
      1) Ingest latest repo + model topology into Graphiti/FalkorDB
      2) Multi-agent ideation
      3) Consensus plan
      4) Developer implements on a new branch
      5) Reviewer gate (iterate if needed)
      6) Open PR
    """

    type: Literal["platform_improve"] = "platform_improve"
    name: str

    # High-level goal for ideation/discussion.
    topic: str = "Propose and implement one high-impact improvement to the Caramba platform."

    # Ingestion controls
    ingest_repo: bool = True
    ingest_models: bool = True
    index_namespace: str = "main"
    ingest_agent: str = "research_team_leader"
    max_files: int = Field(default=250, ge=1)
    max_chars_per_file: int = Field(default=4000, ge=256)

    # Team roles
    leader: str = "research_team_leader"
    ideators: list[str] = Field(default_factory=list)
    developer: str = "developer"
    reviewer: str = "reviewer"

    # Dev/review automation
    repo_root: str = "."
    base_branch: str = "main"
    branch_prefix: str = "agent/platform-improve"
    tests: list[str] = Field(default_factory=lambda: ["python -m pytest -q"])
    max_review_rounds: int = Field(default=2, ge=0)
    open_pr: bool = True
    pr_title_prefix: str = "[caramba] "


AgentProcessConfig: TypeAlias = Annotated[
    DiscussionProcessConfig
    | PaperWriteProcessConfig
    | PaperReviewProcessConfig
    | ResearchLoopProcessConfig
    | CodeGraphSyncProcessConfig
    | PlatformImproveProcessConfig,
    Field(discriminator="type"),
]


class AgentsConfig(BaseModel):
    """Top-level agent section inside a manifest."""

    team: AgentTeamConfig
    processes: list[AgentProcessConfig] = Field(default_factory=list)

