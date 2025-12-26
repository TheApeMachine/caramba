"""Multi-agent research team for collaborative ideation.

This module implements a research collaboration system where multiple
specialized AI agents discuss, debate, and refine research ideas:

- Research Team Leader: Orchestrates discussion, drives consensus
- Domain Expert Agents: Contribute specialized knowledge and critique
- Tools: Search knowledge base, propose experiments, challenge ideas

The system produces structured research proposals with full discussion
transcripts for human review.

Streaming is enabled by default for real-time output of agent responses.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel, Field
from rich.console import Console

from console import logger

if TYPE_CHECKING:
    from config.manifest import Manifest

# Rich console for streaming output
_console = Console()


# ============================================================================
# Data Models
# ============================================================================


class MessageRole(str, Enum):
    """Role of a message in the discussion."""
    LEADER = "leader"
    EXPERT = "expert"
    SYSTEM = "system"
    CONSENSUS = "consensus"


class DiscussionPhase(str, Enum):
    """Phase of the research discussion."""
    OPENING = "opening"           # Leader presents the problem
    IDEATION = "ideation"         # Experts propose ideas
    CRITIQUE = "critique"         # Experts challenge each other
    REFINEMENT = "refinement"     # Refining promising ideas
    SYNTHESIS = "synthesis"       # Combining insights
    CONSENSUS = "consensus"       # Reaching agreement
    CLOSING = "closing"           # Final summary


@dataclass
class Message:
    """A message in the research discussion."""
    role: MessageRole
    agent_name: str
    content: str
    phase: DiscussionPhase
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "agent_name": self.agent_name,
            "content": self.content,
            "phase": self.phase.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ResearchIdea(BaseModel):
    """A research idea proposed during discussion."""
    title: str = Field(description="Short title for the idea")
    description: str = Field(description="Detailed description")
    proposed_by: str = Field(description="Agent who proposed it")
    supporters: list[str] = Field(default_factory=list)
    critiques: list[str] = Field(default_factory=list)
    refinements: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, description="Consensus score 0-10")
    status: str = Field(default="proposed")  # proposed, debated, refined, accepted, rejected


class ResearchProposal(BaseModel):
    """Final research proposal from team consensus."""
    title: str
    summary: str
    key_insights: list[str]
    proposed_experiments: list[dict[str, Any]]
    methodology: str
    expected_outcomes: list[str]
    risks_and_mitigations: list[dict[str, str]]
    next_steps: list[str]
    consensus_score: float
    dissenting_opinions: list[str] = Field(default_factory=list)


class DiscussionTranscript(BaseModel):
    """Complete transcript of the research discussion."""
    session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    manifest_path: str
    topic: str
    phases_completed: list[str] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    ideas_discussed: list[ResearchIdea] = Field(default_factory=list)
    final_proposal: ResearchProposal | None = None
    participating_agents: list[str] = Field(default_factory=list)


# ============================================================================
# Expert Agent Personas
# ============================================================================


EXPERT_PERSONAS = {
    "ml_architect": {
        "name": "Dr. Maya Chen",
        "title": "ML Systems Architect",
        "expertise": ["model architecture", "efficiency", "scaling", "systems design"],
        "personality": "Pragmatic and systems-focused. Asks 'will this actually work at scale?'",
        "instructions": """You are Dr. Maya Chen, an ML Systems Architect with 15 years of experience.

Your expertise: Neural network architectures, efficient inference, scaling laws, systems optimization.

Your role in discussions:
- Evaluate ideas from a systems/architecture perspective
- Ask about computational costs, memory requirements, scaling behavior
- Propose architectural improvements and optimizations
- Challenge ideas that seem impractical to implement
- Share relevant prior work on efficient architectures

Your personality:
- Pragmatic and implementation-focused
- Skeptical of ideas that lack clear implementation paths
- Value simplicity and elegance in design
- Often ask "but how would we actually build this?"

When critiquing: Be constructive but firm. If an idea won't scale, say so clearly.
When proposing: Focus on concrete, implementable solutions.""",
    },

    "theory_expert": {
        "name": "Prof. James Morrison",
        "title": "Theoretical ML Researcher",
        "expertise": ["learning theory", "optimization", "generalization", "mathematical foundations"],
        "personality": "Rigorous and analytical. Demands mathematical justification.",
        "instructions": """You are Prof. James Morrison, a theoretical ML researcher.

Your expertise: Learning theory, optimization, generalization bounds, mathematical foundations of ML.

Your role in discussions:
- Provide theoretical grounding for proposed ideas
- Identify potential theoretical issues or impossibility results
- Suggest connections to existing theoretical frameworks
- Challenge hand-wavy arguments that lack rigor
- Propose theoretically-motivated improvements

Your personality:
- Rigorous and analytical
- Demand clear definitions and formal statements
- Value theoretical elegance
- Often ask "what's the formal guarantee here?"

When critiquing: Point out logical gaps and missing assumptions.
When proposing: Ground ideas in established theory.""",
    },

    "experimentalist": {
        "name": "Dr. Sarah Park",
        "title": "Experimental ML Scientist",
        "expertise": ["experiment design", "benchmarking", "ablations", "reproducibility"],
        "personality": "Empirically-driven. 'Show me the numbers.'",
        "instructions": """You are Dr. Sarah Park, an experimental ML scientist.

Your expertise: Experiment design, benchmarking, ablation studies, statistical analysis, reproducibility.

Your role in discussions:
- Propose concrete experiments to test ideas
- Identify confounds and potential experimental issues
- Suggest appropriate baselines and metrics
- Challenge claims that lack empirical support
- Design ablation studies to isolate effects

Your personality:
- Empirically-driven and data-focused
- Skeptical of claims without experimental backing
- Value reproducibility and fair comparisons
- Often ask "how would we measure this?"

When critiquing: Demand experimental evidence for claims.
When proposing: Include concrete experimental protocols.""",
    },

    "applications_expert": {
        "name": "Dr. Alex Rivera",
        "title": "Applied ML Engineer",
        "expertise": ["deployment", "real-world constraints", "user needs", "practical trade-offs"],
        "personality": "User-focused and practical. 'Will this help real users?'",
        "instructions": """You are Dr. Alex Rivera, an applied ML engineer.

Your expertise: Model deployment, production constraints, user requirements, practical trade-offs.

Your role in discussions:
- Ground discussions in real-world use cases
- Identify practical constraints and deployment challenges
- Advocate for user needs and practical utility
- Challenge ideas that seem academically interesting but impractical
- Propose pragmatic solutions that balance idealism with reality

Your personality:
- User-focused and practical
- Value solutions that ship and work in production
- Often bridge research and engineering perspectives
- Ask "what problem does this actually solve?"

When critiquing: Challenge ivory-tower thinking.
When proposing: Focus on high-impact, deployable solutions.""",
    },

    "domain_specialist": {
        "name": "Dr. Wei Zhang",
        "title": "NLP/Language Model Specialist",
        "expertise": ["language models", "attention mechanisms", "transformers", "NLP"],
        "personality": "Deep domain knowledge. Connects ideas to LLM literature.",
        "instructions": """You are Dr. Wei Zhang, an NLP and language model specialist.

Your expertise: Transformers, attention mechanisms, language modeling, efficient inference for LLMs.

Your role in discussions:
- Provide deep expertise on transformer architectures
- Connect ideas to relevant NLP/LLM literature
- Identify domain-specific challenges and opportunities
- Suggest improvements based on recent advances
- Challenge ideas that misunderstand how LLMs work

Your personality:
- Deep domain expertise
- Stay current with latest research
- Value techniques that work well for language tasks
- Often reference specific papers and methods

When critiquing: Point out domain-specific issues others might miss.
When proposing: Draw on cutting-edge NLP research.""",
    },
}


LEADER_INSTRUCTIONS = """You are the Research Team Leader orchestrating a collaborative research discussion.

## Your Role

You lead a team of expert researchers in evaluating and refining research directions. Your job is to:

1. **Facilitate productive discussion** - Keep the conversation focused and moving forward
2. **Ensure all voices are heard** - Draw out insights from each expert
3. **Drive toward consensus** - Help the team converge on actionable conclusions
4. **Synthesize insights** - Combine diverse perspectives into coherent proposals

## Discussion Phases

You guide the team through these phases:

1. **OPENING**: Present the research problem/context from the manifest
2. **IDEATION**: Ask each expert to propose ideas from their perspective
3. **CRITIQUE**: Have experts challenge and question each other's ideas
4. **REFINEMENT**: Guide refinement of promising ideas based on critiques
5. **SYNTHESIS**: Combine insights into coherent proposals
6. **CONSENSUS**: Reach agreement on final recommendations
7. **CLOSING**: Summarize conclusions and next steps

## Your Tools

- **search_knowledge_base**: Search previously discovered papers
- **search_arxiv**: Search for new papers on arXiv
- **propose_research_direction**: Formally propose a direction for team consideration
- **call_for_critique**: Ask experts to critique a specific idea
- **call_for_vote**: Ask experts to vote on a proposal
- **declare_consensus**: Declare that consensus has been reached
- **generate_proposal**: Generate the final research proposal artifact

## Guidelines

- Be a fair moderator - don't favor any expert's views
- Push back when discussion becomes circular
- Identify when enough debate has occurred
- Keep track of the best ideas emerging from discussion
- Be decisive when it's time to move forward
- Document key decision points and reasoning

## Output Format

When speaking, use this format:
[PHASE: current_phase]
[TO: @all or @specific_expert]
Your message here...
"""


# ============================================================================
# Research Team
# ============================================================================


class ResearchTeam:
    """Orchestrates multi-agent research discussions.

    Creates a team of specialized AI agents who discuss, debate, and
    refine research ideas, producing structured proposals for human review.

    Usage:
        team = ResearchTeam(manifest_path="config/presets/my_experiment.yml")
        result = await team.run_session(topic="Improving KV-cache efficiency")
    """

    def __init__(
        self,
        manifest_path: Path | str,
        output_dir: Path | str | None = None,
        experts: list[str] | None = None,
        max_rounds: int = 10,
        model: str = "gpt-4o",
    ) -> None:
        """Initialize the research team.

        Args:
            manifest_path: Path to the experiment manifest.
            output_dir: Directory for output artifacts.
            experts: List of expert persona keys to include (default: all).
            max_rounds: Maximum discussion rounds before forcing consensus.
            model: Model to use for all agents.
        """
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir) if output_dir else Path("artifacts/research")
        self.max_rounds = max_rounds
        self.model = model

        # Select experts
        if experts is None:
            experts = list(EXPERT_PERSONAS.keys())
        self.expert_keys = experts

        # State
        self.messages: list[Message] = []
        self.ideas: list[ResearchIdea] = []
        self.current_phase = DiscussionPhase.OPENING
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Agents (lazy-loaded)
        self._leader_agent: Any = None
        self._expert_agents: dict[str, Any] = {}

    def _create_agents(self) -> None:
        """Create the leader and expert agents."""
        try:
            from agents import Agent
        except ImportError:
            raise ImportError("openai-agents not installed. Run: pip install openai-agents")

        # Create leader agent
        self._leader_agent = Agent(
            name="Research Team Leader",
            instructions=LEADER_INSTRUCTIONS,
            model=self.model,
        )

        # Create expert agents
        for key in self.expert_keys:
            persona = EXPERT_PERSONAS.get(key)
            if persona:
                self._expert_agents[key] = Agent(
                    name=persona["name"],
                    instructions=persona["instructions"],
                    model=self.model,
                )

        logger.info(f"Created research team with {len(self._expert_agents)} experts")

    async def run_session(
        self,
        topic: str | None = None,
        context: str | None = None,
    ) -> DiscussionTranscript:
        """Run a complete research discussion session.

        Args:
            topic: The research topic/question to discuss.
            context: Additional context (e.g., experiment goals).

        Returns:
            Complete transcript including final proposal.
        """
        from agents import Runner

        self._create_agents()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest for context
        manifest_content = ""
        if self.manifest_path.exists():
            manifest_content = self.manifest_path.read_text()

        if topic is None:
            topic = "Research directions for the experiment defined in the manifest"

        logger.header("Research Team Session", topic)
        logger.key_value({
            "Session ID": self.session_id,
            "Experts": ", ".join(EXPERT_PERSONAS[k]["name"] for k in self.expert_keys),
            "Max Rounds": self.max_rounds,
        })

        # Initialize transcript
        transcript = DiscussionTranscript(
            session_id=self.session_id,
            started_at=datetime.now(),
            manifest_path=str(self.manifest_path),
            topic=topic,
            participating_agents=["Research Team Leader"] + [
                EXPERT_PERSONAS[k]["name"] for k in self.expert_keys
            ],
        )

        # Run discussion phases
        try:
            await self._run_opening(topic, manifest_content, context, transcript)
            await self._run_ideation(transcript)
            await self._run_critique(transcript)
            await self._run_refinement(transcript)
            await self._run_synthesis(transcript)
            await self._run_consensus(transcript)
            await self._run_closing(transcript)

        except Exception as e:
            logger.error(f"Session error: {e}")
            self._add_message(
                MessageRole.SYSTEM,
                "System",
                f"Session ended due to error: {e}",
                DiscussionPhase.CLOSING,
            )

        # Finalize transcript
        transcript.ended_at = datetime.now()
        transcript.messages = [m.to_dict() for m in self.messages]
        transcript.ideas_discussed = self.ideas

        # Save artifacts
        await self._save_artifacts(transcript)

        return transcript

    async def _run_opening(
        self,
        topic: str,
        manifest_content: str,
        context: str | None,
        transcript: DiscussionTranscript,
    ) -> None:
        """Opening phase: Leader presents the problem."""
        self.current_phase = DiscussionPhase.OPENING
        transcript.phases_completed.append("opening")
        logger.subheader("Phase: Opening")

        prompt = f"""We're starting a research discussion session.

## Topic
{topic}

## Experiment Context
```yaml
{manifest_content[:2000]}...
```

{f"## Additional Context" + chr(10) + context if context else ""}

Please open the discussion by:
1. Summarizing the research problem
2. Highlighting key challenges and opportunities
3. Setting the stage for expert ideation

Address all experts (@all) and invite them to share their initial thoughts."""

        response = await self._get_leader_response(prompt)
        self._add_message(MessageRole.LEADER, "Research Team Leader", response, self.current_phase)

    async def _run_ideation(self, transcript: DiscussionTranscript) -> None:
        """Ideation phase: Experts propose ideas."""
        self.current_phase = DiscussionPhase.IDEATION
        transcript.phases_completed.append("ideation")
        logger.subheader("Phase: Ideation")

        # Get each expert's initial ideas
        for key in self.expert_keys:
            persona = EXPERT_PERSONAS[key]
            prompt = f"""The Research Team Leader has opened our discussion on the topic.

Based on your expertise in {', '.join(persona['expertise'])}, please:
1. Propose 1-2 concrete research directions or ideas
2. Explain the rationale from your perspective
3. Identify potential challenges you foresee

Discussion so far:
{self._get_recent_context(5)}"""

            response = await self._get_expert_response(key, prompt)
            self._add_message(MessageRole.EXPERT, persona["name"], response, self.current_phase)

            # Extract and track ideas
            self._extract_ideas(response, persona["name"])

    async def _run_critique(self, transcript: DiscussionTranscript) -> None:
        """Critique phase: Experts challenge each other."""
        self.current_phase = DiscussionPhase.CRITIQUE
        transcript.phases_completed.append("critique")
        logger.subheader("Phase: Critique")

        # Each expert critiques others' ideas
        for key in self.expert_keys:
            persona = EXPERT_PERSONAS[key]
            prompt = f"""We're now in the critique phase. Review the ideas proposed by other experts.

From your perspective as {persona['title']}:
1. What concerns or weaknesses do you see in the proposed ideas?
2. What questions would you want answered before proceeding?
3. Which ideas seem most promising and why?

Be constructive but rigorous. Challenge assumptions that seem shaky.

Ideas proposed so far:
{self._format_ideas()}

Recent discussion:
{self._get_recent_context(5)}"""

            response = await self._get_expert_response(key, prompt)
            self._add_message(MessageRole.EXPERT, persona["name"], response, self.current_phase)

            # Update ideas with critiques
            self._add_critiques(response, persona["name"])

    async def _run_refinement(self, transcript: DiscussionTranscript) -> None:
        """Refinement phase: Improve ideas based on critiques."""
        self.current_phase = DiscussionPhase.REFINEMENT
        transcript.phases_completed.append("refinement")
        logger.subheader("Phase: Refinement")

        # Leader summarizes critiques and asks for refinements
        leader_prompt = f"""We've heard critiques from all experts. Please:
1. Summarize the key concerns raised
2. Identify which ideas survived critique best
3. Ask specific experts to refine their proposals addressing the critiques

Discussion so far:
{self._get_recent_context(10)}"""

        leader_response = await self._get_leader_response(leader_prompt)
        self._add_message(MessageRole.LEADER, "Research Team Leader", leader_response, self.current_phase)

        # Experts refine their ideas
        for key in self.expert_keys[:3]:  # Top 3 experts refine
            persona = EXPERT_PERSONAS[key]
            prompt = f"""Based on the critiques and leader's feedback, please refine your proposals:
1. Address the specific concerns raised about your ideas
2. Incorporate insights from other experts
3. Present improved versions of your most promising ideas

Leader's guidance:
{leader_response[:500]}

Your original ideas and the critiques:
{self._get_expert_ideas(persona["name"])}"""

            response = await self._get_expert_response(key, prompt)
            self._add_message(MessageRole.EXPERT, persona["name"], response, self.current_phase)

    async def _run_synthesis(self, transcript: DiscussionTranscript) -> None:
        """Synthesis phase: Combine insights into proposals."""
        self.current_phase = DiscussionPhase.SYNTHESIS
        transcript.phases_completed.append("synthesis")
        logger.subheader("Phase: Synthesis")

        prompt = f"""We're now synthesizing our discussion into concrete proposals.

Please create 1-2 unified research proposals that:
1. Combine the best ideas from multiple experts
2. Address the major critiques raised
3. Include concrete next steps and experiments
4. Acknowledge remaining uncertainties

Full discussion:
{self._get_full_context()}

Ideas discussed:
{self._format_ideas()}"""

        response = await self._get_leader_response(prompt)
        self._add_message(MessageRole.LEADER, "Research Team Leader", response, self.current_phase)

    async def _run_consensus(self, transcript: DiscussionTranscript) -> None:
        """Consensus phase: Reach agreement."""
        self.current_phase = DiscussionPhase.CONSENSUS
        transcript.phases_completed.append("consensus")
        logger.subheader("Phase: Consensus")

        # Quick vote from each expert
        votes: dict[str, str] = {}
        for key in self.expert_keys:
            persona = EXPERT_PERSONAS[key]
            prompt = f"""The leader has synthesized our proposals. Please provide:
1. Your support level (STRONG SUPPORT / SUPPORT / NEUTRAL / CONCERNS / OPPOSE)
2. Brief rationale (1-2 sentences)
3. Any final reservations to note

Synthesized proposal:
{self._get_recent_context(3)}"""

            response = await self._get_expert_response(key, prompt)
            self._add_message(MessageRole.EXPERT, persona["name"], response, self.current_phase)
            votes[persona["name"]] = response

        # Leader declares consensus
        consensus_prompt = f"""All experts have voted. Please:
1. Summarize the consensus level
2. Note any dissenting opinions
3. Declare the final decision

Votes:
{json.dumps(votes, indent=2)}"""

        response = await self._get_leader_response(consensus_prompt)
        self._add_message(MessageRole.CONSENSUS, "Research Team Leader", response, self.current_phase)

    async def _run_closing(self, transcript: DiscussionTranscript) -> None:
        """Closing phase: Generate final proposal."""
        self.current_phase = DiscussionPhase.CLOSING
        transcript.phases_completed.append("closing")
        logger.subheader("Phase: Closing")

        prompt = f"""Please generate the final research proposal in this exact JSON format:
{{
    "title": "Proposal title",
    "summary": "2-3 paragraph summary",
    "key_insights": ["insight 1", "insight 2", ...],
    "proposed_experiments": [
        {{"name": "exp1", "description": "...", "expected_outcome": "..."}},
        ...
    ],
    "methodology": "Detailed methodology description",
    "expected_outcomes": ["outcome 1", "outcome 2", ...],
    "risks_and_mitigations": [
        {{"risk": "...", "mitigation": "..."}},
        ...
    ],
    "next_steps": ["step 1", "step 2", ...],
    "consensus_score": 8.5,
    "dissenting_opinions": ["any noted dissent"]
}}

Base this on our full discussion:
{self._get_full_context()}"""

        response = await self._get_leader_response(prompt)

        # Parse the proposal
        try:
            # Extract JSON from response
            json_match = response[response.find("{"):response.rfind("}")+1]
            proposal_data = json.loads(json_match)
            transcript.final_proposal = ResearchProposal(**proposal_data)
            logger.success("Final proposal generated")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Could not parse proposal JSON: {e}")
            # Create a basic proposal from the response
            transcript.final_proposal = ResearchProposal(
                title="Research Proposal",
                summary=response[:500],
                key_insights=[],
                proposed_experiments=[],
                methodology="See discussion transcript",
                expected_outcomes=[],
                risks_and_mitigations=[],
                next_steps=[],
                consensus_score=0.0,
            )

    async def _get_leader_response(self, prompt: str) -> str:
        """Get a streaming response from the leader agent."""
        return await self._stream_agent_response(
            self._leader_agent,
            prompt,
            agent_name="Research Team Leader",
            style="bold cyan",
        )

    async def _get_expert_response(self, expert_key: str, prompt: str) -> str:
        """Get a streaming response from an expert agent."""
        agent = self._expert_agents.get(expert_key)
        if not agent:
            return f"[Expert {expert_key} not available]"

        persona = EXPERT_PERSONAS[expert_key]
        return await self._stream_agent_response(
            agent,
            prompt,
            agent_name=persona["name"],
            style="bold green",
        )

    async def _stream_agent_response(
        self,
        agent: Any,
        prompt: str,
        agent_name: str,
        style: str = "bold white",
    ) -> str:
        """Stream an agent response with real-time output.

        This provides immediate feedback as the agent generates its response,
        including tool call notifications and reasoning content.
        """
        from agents import Runner

        # Print agent header
        _console.print(f"\n[{style}]{agent_name}:[/{style}] ", end="")

        full_response = ""
        current_tool = None

        # Use streaming runner
        result = Runner.run_streamed(agent, input=prompt)

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                # Stream text deltas
                if isinstance(event.data, ResponseTextDeltaEvent):
                    delta = event.data.delta
                    _console.print(delta, end="", markup=False)
                    full_response += delta

                # Handle reasoning content (yellow)
                elif event.data.type == "response.reasoning_summary_text.delta":
                    delta = getattr(event.data, "delta", "")
                    _console.print(f"[dim italic]{delta}[/dim italic]", end="")

                # Tool call started
                elif event.data.type == "response.output_item.added":
                    if getattr(event.data.item, "type", None) == "function_call":
                        tool_name = getattr(event.data.item, "name", "unknown")
                        current_tool = tool_name
                        _console.print(
                            f"\n  [dim]→ Using tool: {tool_name}[/dim]",
                            end="",
                        )

                # Tool call completed
                elif event.data.type == "response.output_item.done":
                    if current_tool:
                        _console.print(" [dim]✓[/dim]")
                        current_tool = None

            elif event.type == "run_item_stream_event":
                # Tool output (don't print the actual output, just acknowledge)
                if event.item.type == "tool_call_output_item":
                    pass  # Tool results are internal, don't display

        _console.print()  # Final newline
        return full_response

    def _add_message(
        self,
        role: MessageRole,
        agent_name: str,
        content: str,
        phase: DiscussionPhase,
    ) -> None:
        """Add a message to the discussion."""
        self.messages.append(Message(
            role=role,
            agent_name=agent_name,
            content=content,
            phase=phase,
        ))

    def _get_recent_context(self, n: int = 5) -> str:
        """Get the last n messages as context."""
        recent = self.messages[-n:] if len(self.messages) >= n else self.messages
        return "\n\n".join(
            f"**{m.agent_name}** ({m.phase.value}):\n{m.content}"
            for m in recent
        )

    def _get_full_context(self) -> str:
        """Get the full discussion context."""
        return self._get_recent_context(len(self.messages))

    def _extract_ideas(self, response: str, proposer: str) -> None:
        """Extract research ideas from a response."""
        # Simple extraction - in practice, could use structured output
        idea = ResearchIdea(
            title=f"Idea from {proposer}",
            description=response[:500],
            proposed_by=proposer,
        )
        self.ideas.append(idea)

    def _add_critiques(self, response: str, critic: str) -> None:
        """Add critiques to existing ideas."""
        for idea in self.ideas:
            if idea.proposed_by != critic:
                idea.critiques.append(f"{critic}: {response[:200]}")

    def _format_ideas(self) -> str:
        """Format ideas for display."""
        if not self.ideas:
            return "No ideas proposed yet."
        return "\n\n".join(
            f"**{idea.title}** (by {idea.proposed_by}):\n{idea.description[:300]}..."
            for idea in self.ideas
        )

    def _get_expert_ideas(self, expert_name: str) -> str:
        """Get ideas proposed by a specific expert."""
        expert_ideas = [i for i in self.ideas if i.proposed_by == expert_name]
        if not expert_ideas:
            return "No ideas from this expert."
        return "\n\n".join(
            f"**{idea.title}**:\n{idea.description}\nCritiques: {idea.critiques}"
            for idea in expert_ideas
        )

    async def _save_artifacts(self, transcript: DiscussionTranscript) -> None:
        """Save discussion artifacts."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save full transcript
        transcript_path = self.output_dir / f"transcript_{self.session_id}.json"
        transcript_path.write_text(
            transcript.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.path(str(transcript_path), "Transcript saved")

        # Save proposal as markdown
        if transcript.final_proposal:
            proposal_path = self.output_dir / f"proposal_{self.session_id}.md"
            proposal_md = self._format_proposal_markdown(transcript.final_proposal)
            proposal_path.write_text(proposal_md, encoding="utf-8")
            logger.path(str(proposal_path), "Proposal saved")

        # Save discussion summary
        summary_path = self.output_dir / f"summary_{self.session_id}.md"
        summary_md = self._format_summary_markdown(transcript)
        summary_path.write_text(summary_md, encoding="utf-8")
        logger.path(str(summary_path), "Summary saved")

    def _format_proposal_markdown(self, proposal: ResearchProposal) -> str:
        """Format proposal as markdown."""
        experiments = "\n".join(
            f"- **{e.get('name', 'Unnamed')}**: {e.get('description', '')}"
            for e in proposal.proposed_experiments
        )
        risks = "\n".join(
            f"- **{r.get('risk', '')}**: {r.get('mitigation', '')}"
            for r in proposal.risks_and_mitigations
        )

        return f"""# {proposal.title}

**Consensus Score: {proposal.consensus_score}/10**

## Summary

{proposal.summary}

## Key Insights

{chr(10).join(f"- {i}" for i in proposal.key_insights)}

## Proposed Experiments

{experiments}

## Methodology

{proposal.methodology}

## Expected Outcomes

{chr(10).join(f"- {o}" for o in proposal.expected_outcomes)}

## Risks and Mitigations

{risks}

## Next Steps

{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(proposal.next_steps))}

## Dissenting Opinions

{chr(10).join(f"- {d}" for d in proposal.dissenting_opinions) if proposal.dissenting_opinions else "None noted."}
"""

    def _format_summary_markdown(self, transcript: DiscussionTranscript) -> str:
        """Format discussion summary as markdown."""
        return f"""# Research Discussion Summary

**Session ID**: {transcript.session_id}
**Topic**: {transcript.topic}
**Started**: {transcript.started_at}
**Ended**: {transcript.ended_at}

## Participants

{chr(10).join(f"- {p}" for p in transcript.participating_agents)}

## Phases Completed

{chr(10).join(f"- {p}" for p in transcript.phases_completed)}

## Discussion Highlights

{self._get_full_context()[:5000]}...

## Ideas Discussed

{self._format_ideas()}
"""

    def run_sync(
        self,
        topic: str | None = None,
        context: str | None = None,
    ) -> DiscussionTranscript:
        """Synchronous wrapper for run_session."""
        return asyncio.run(self.run_session(topic=topic, context=context))


# ============================================================================
# Convenience Functions
# ============================================================================


async def run_research_session(
    manifest_path: Path | str,
    topic: str | None = None,
    output_dir: Path | str | None = None,
    experts: list[str] | None = None,
    model: str = "gpt-4o",
) -> DiscussionTranscript:
    """Run a research team discussion session.

    Args:
        manifest_path: Path to the experiment manifest.
        topic: Research topic to discuss.
        output_dir: Output directory for artifacts.
        experts: List of expert keys to include.
        model: Model to use for agents.

    Returns:
        Complete discussion transcript with proposal.
    """
    team = ResearchTeam(
        manifest_path=manifest_path,
        output_dir=output_dir,
        experts=experts,
        model=model,
    )
    return await team.run_session(topic=topic)


def run_research_session_sync(
    manifest_path: Path | str,
    topic: str | None = None,
    output_dir: Path | str | None = None,
    experts: list[str] | None = None,
    model: str = "gpt-4o",
) -> DiscussionTranscript:
    """Synchronous version of run_research_session."""
    return asyncio.run(run_research_session(
        manifest_path=manifest_path,
        topic=topic,
        output_dir=output_dir,
        experts=experts,
        model=model,
    ))
