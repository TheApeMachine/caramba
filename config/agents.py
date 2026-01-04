"""Agent process configuration.

This extends the manifest-driven approach to the built-in agent system:
- define a *team* (keys -> persona yaml names)
- define one or more *processes* (discussion, paper workflows, etc.)
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, RootModel, model_validator

from .platform_improve import PlatformImproveProcessConfig

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
    name: str | None = None
    leader: str
    topic: str
    prompts_dir: str = "config/prompts"
    max_rounds: int = Field(default=12, ge=1)

    @model_validator(mode="after")
    def _default_name(self) -> "DiscussionProcessConfig":
        if not self.name:
            self.name = self.type
        return self


class PaperWriteProcessConfig(BaseModel):
    """Draft/update a paper artifact using a writer persona."""

    type: Literal["paper_write"] = "paper_write"
    name: str | None = None
    writer: str
    # Optional: high-level goal/instructions for the writer.
    goal: str = ""
    # Where to read/write paper artifacts (usually under artifacts/).
    output_dir: str = "paper"

    @model_validator(mode="after")
    def _default_name(self) -> "PaperWriteProcessConfig":
        if not self.name:
            self.name = self.type
        return self


class PaperReviewProcessConfig(BaseModel):
    """Review a paper artifact using a reviewer persona."""

    type: Literal["paper_review"] = "paper_review"
    name: str | None = None
    reviewer: str
    strictness: str = "conference"
    max_proposed_experiments: int = Field(default=3, ge=0)
    # Optional reviewer-specific instructions to append.
    goal: str = ""

    @model_validator(mode="after")
    def _default_name(self) -> "PaperReviewProcessConfig":
        if not self.name:
            self.name = self.type
        return self


class ResearchLoopProcessConfig(BaseModel):
    """Orchestrate write → review → structural audit → (optionally) experiments."""

    type: Literal["research_loop"] = "research_loop"
    name: str | None = None
    leader: str
    writer: str
    reviewer: str
    max_iterations: int = Field(default=5, ge=1)
    # If true, allow the loop to run experiments automatically (Phase 2 readiness applies).
    auto_run_experiments: bool = False
    output_dir: str = "paper"

    @model_validator(mode="after")
    def _default_name(self) -> "ResearchLoopProcessConfig":
        if not self.name:
            self.name = self.type
        return self


class IdleProcessConfig(BaseModel):
    """Budgeted idle loop.

    This process is designed to run when the system is "idle" and convert spare
    cycles into durable, diffable artifacts:
      - readiness work (e.g. code_graph_sync)
      - quick evaluation commands (e.g. unit tests, short benchmarks)
      - a short research loop iteration (paper_write → paper_review → audit)

    All steps are optional and governed by a wall-clock budget.
    """

    type: Literal["idle"] = "idle"
    name: str | None = None

    # Overall wall-clock budget for the idle run.
    max_wall_time_sec: int = Field(default=600, ge=1)

    # --- Readiness ---
    run_code_graph_sync: bool = True
    code_graph_sync_agent: str = "leader"
    index_namespace: str = "main"

    # --- Evaluation ---
    run_eval: bool = False
    eval_cmds: list[str] = Field(default_factory=list)
    eval_timeout_sec: int = Field(default=300, ge=1)
    eval_cwd: str = "."

    # --- Research loop ---
    run_research_loop: bool = True
    leader: str = "leader"
    writer: str = "writer"
    reviewer: str = "reviewer"
    research_max_iterations: int = Field(default=1, ge=1)
    research_auto_run_experiments: bool = False
    output_dir: str = "paper"

    @model_validator(mode="after")
    def _default_name(self) -> "IdleProcessConfig":
        if not self.name:
            self.name = self.type
        return self


class CodeGraphSyncProcessConfig(BaseModel):
    """Ingest manifest/model topology into Graphiti graph memory."""

    type: Literal["code_graph_sync"] = "code_graph_sync"
    name: str | None = None
    agent: str
    # Prefix/namespace to avoid collisions across manifests/runs.
    index_namespace: str = "main"

    @model_validator(mode="after")
    def _default_name(self) -> "CodeGraphSyncProcessConfig":
        if not self.name:
            self.name = self.type
        return self


class PaperCollectArtifactsProcessConfig(BaseModel):
    """Collect benchmark artifacts into paper-ready tables/figures.

    This is a non-LLM utility process: it does not require agents or MCP tools.
    It exists so paper workflows remain manifest-driven end-to-end.
    """

    type: Literal["paper_collect_artifacts"] = "paper_collect_artifacts"
    name: str | None = None
    # Where the platform writes per-target artifacts (default: ./artifacts).
    artifact_root: str = "artifacts"
    # Where to write paper-ready outputs (default: ./artifacts/paper).
    out_dir: str = "artifacts/paper"
    # Caption/title prefix used in generated LaTeX and figures.
    title: str = "DBA Ablations"
    # Optional explicit target list (defaults to all experiment targets in the manifest).
    targets: list[str] | None = None

    @model_validator(mode="after")
    def _default_name(self) -> "PaperCollectArtifactsProcessConfig":
        if not self.name:
            self.name = self.type
        return self


class MultiplexChatProcessConfig(BaseModel):
    """Interactive multiplex chat REPL across multiple model-backed agents."""

    type: Literal["multiplex_chat"] = "multiplex_chat"
    name: str | None = None

    # Tag -> agent key mapping (keys are the @tags, values are keys in agents.team).
    routes: dict[str, str] = Field(
        default_factory=lambda: {"chatgpt": "chatgpt", "claude": "claude", "gemini": "gemini"}
    )
    initial_route: str = "chatgpt"

    # Output controls.
    stream: bool = True
    show_reasoning: bool = False
    show_output: bool = True

    # Display name for the human user in the transcript.
    user_name: str = "user"

    # Transcript and prompt budgeting (manifest-driven; no environment variables).
    transcript_path: str = "artifacts/ai/brainstorm.jsonl"
    max_context_items: int = Field(default=40, ge=1)
    # Transcript budgeting shared across all providers in the multiplex chat.
    # Keep this below the smallest provider context window and leave headroom for
    # system prompts + tool schemas + tool results.
    max_context_tokens: int = Field(default=128000, ge=1024)
    max_event_tokens: int = Field(default=8192, ge=256)
    compact_after_bytes: int = Field(default=2_000_000, ge=1024)

    @model_validator(mode="after")
    def _default_name(self) -> "MultiplexChatProcessConfig":
        if not self.name:
            self.name = self.type
        return self


AgentProcessConfig: TypeAlias = Annotated[
    DiscussionProcessConfig
    | PaperWriteProcessConfig
    | PaperReviewProcessConfig
    | ResearchLoopProcessConfig
    | IdleProcessConfig
    | CodeGraphSyncProcessConfig
    | PlatformImproveProcessConfig
    | PaperCollectArtifactsProcessConfig
    | MultiplexChatProcessConfig,
    Field(discriminator="type"),
]


class AgentsConfig(BaseModel):
    """Top-level agent section inside a manifest."""

    team: AgentTeamConfig
    processes: list[AgentProcessConfig] = Field(default_factory=list)

