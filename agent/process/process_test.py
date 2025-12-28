from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from agent import Researcher
from agent.process import Process


@dataclass
class DummyAgent:
    handoffs: list | None = None


@dataclass
class DummyResearcher:
    agent: DummyAgent


class DummyProcess(Process):
    pass


def test_process_next_agent_returns_by_key() -> None:
    agents_dict = {
        "a": cast(Researcher, DummyResearcher(DummyAgent())),
        "b": cast(Researcher, DummyResearcher(DummyAgent())),
    }
    p = DummyProcess(agents_dict)
    assert p.next_agent("a") is p.agents["a"]


def test_process_handoff_appends_handoff_entry(monkeypatch) -> None:
    agents_dict = {
        "a": cast(Researcher, DummyResearcher(DummyAgent())),
        "b": cast(Researcher, DummyResearcher(DummyAgent())),
    }
    p = DummyProcess(agents_dict)

    # The Process module imports `handoff` directly, so patch that symbol too.
    import agent.process as proc_mod

    monkeypatch.setattr(proc_mod, "handoff", lambda target, input_filter=None: {"to": target, "filter": input_filter})

    p.handoff("a", "b")
    assert p.agents["a"].agent.handoffs is not None
    assert len(p.agents["a"].agent.handoffs) == 1

