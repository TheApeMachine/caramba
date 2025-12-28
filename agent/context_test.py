from __future__ import annotations

from agent.context import AgentContext
from agent.knowledge import Knowledge
from agent.message import Message


def test_agent_context_to_prompt_empty() -> None:
    ctx = AgentContext()
    assert ctx.to_prompt() == ""


def test_agent_context_to_prompt_includes_history_and_knowledge() -> None:
    ctx = AgentContext()
    ctx.add_message(Message(name="User", role="user", content="Hello"))
    ctx.knowledge.append(Knowledge(name="Doc", source="paper", content="Key result"))

    prompt = ctx.to_prompt()
    assert "<history>" in prompt
    assert "**User** (user):" in prompt
    assert "Hello" in prompt
    assert "</history>" in prompt

    assert "<knowledge>" in prompt
    assert "**Doc** (paper):" in prompt
    assert "Key result" in prompt
    assert "</knowledge>" in prompt

