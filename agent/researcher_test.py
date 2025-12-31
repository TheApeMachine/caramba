from __future__ import annotations

import asyncio
import types

import caramba.agent as agent_mod
from caramba.agent import Researcher
from caramba.agent.context import AgentContext
from caramba.config.persona import DeveloperConfig


def _make_researcher(monkeypatch) -> Researcher:
    class FakeAgent:
        def __init__(self, *args, **kwargs) -> None:
            _ = args
            _ = kwargs

    class FakeRunner:
        @staticmethod
        async def run(_agent, *, input: str, context: dict):
            return types.SimpleNamespace(input=input, context=context)

    monkeypatch.setattr(agent_mod, "Agent", FakeAgent)
    monkeypatch.setattr(agent_mod, "Runner", FakeRunner)
    monkeypatch.setattr(agent_mod, "ModelSettings", lambda **kwargs: kwargs)

    persona = DeveloperConfig(
        name="Dev",
        description="dev",
        instructions="do stuff",
        model="gpt-4.1-mini",
        temperature=0.0,
        tool_choice="auto",
        mcp_servers=[],
    )
    return Researcher(persona)


def test_prepare_message_and_ctx_passthrough_dict(monkeypatch) -> None:
    r = _make_researcher(monkeypatch)
    res = asyncio.run(r.run("hello", {"x": 1}))
    assert res.input == "hello"
    assert res.context == {"x": 1}


def test_prepare_message_and_ctx_folds_agent_context_into_message(monkeypatch) -> None:
    r = _make_researcher(monkeypatch)
    ac = AgentContext()
    res = asyncio.run(r.run("hello", ac))
    assert "hello" in res.input
    assert res.context == {}


def test_prepare_message_and_ctx_handles_none_context(monkeypatch) -> None:
    r = _make_researcher(monkeypatch)
    res = asyncio.run(r.run("hello", None))
    assert res.input == "hello"
    assert res.context == {}

