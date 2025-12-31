from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from caramba.agent import Researcher
from caramba.agent.context import AgentContext
from caramba.agent.process import Process


@dataclass
class DummyAgent:
    handoffs: list[object] | None = None


class DummyResearcher(Researcher):
    def __init__(self, agent: DummyAgent) -> None:
        # Do not call Researcher.__init__ (avoids SDK/network wiring in tests).
        self.agent = agent
        self.persona = object()
        self.logger = object()

    async def run(self, message: str, context: dict[str, Any] | AgentContext | None = None) -> Any:
        _ = message
        _ = context
        return None

    async def run_streamed_to_console(
        self,
        message: str,
        context: dict[str, Any] | AgentContext | None = None,
        *,
        show_reasoning: bool = True,
        show_output: bool = True,
    ) -> Any:
        _ = message
        _ = context
        _ = show_reasoning
        _ = show_output
        return None


class DummyProcess(Process):
    pass


def test_process_next_agent_returns_by_key() -> None:
    agents_dict: dict[str, Researcher] = {"a": DummyResearcher(DummyAgent()), "b": DummyResearcher(DummyAgent())}
    p = DummyProcess(agents_dict)
    assert p.next_agent("a") is p.agents["a"]


def test_process_handoff_appends_handoff_entry(monkeypatch) -> None:
    agents_dict: dict[str, Researcher] = {"a": DummyResearcher(DummyAgent()), "b": DummyResearcher(DummyAgent())}
    p = DummyProcess(agents_dict)

    # The Process module imports `handoff` directly, so patch that symbol too.
    import caramba.agent.process as proc_mod

    monkeypatch.setattr(proc_mod, "handoff", lambda target, input_filter=None: {"to": target, "filter": input_filter})

    p.handoff("a", "b")
    assert p.agents["a"].agent.handoffs is not None
    assert len(p.agents["a"].agent.handoffs) == 1

