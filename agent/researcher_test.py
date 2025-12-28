from __future__ import annotations

from agent import Researcher
from agent.context import AgentContext


def test_prepare_message_and_ctx_passthrough_dict() -> None:
    msg, ctx = Researcher._prepare_message_and_ctx({"x": 1}, "hello")
    assert msg == "hello"
    assert ctx == {"x": 1}


def test_prepare_message_and_ctx_folds_agent_context_into_message() -> None:
    ac = AgentContext()
    msg, ctx = Researcher._prepare_message_and_ctx(ac, "hello")
    assert "hello" in msg
    assert ctx == {}


def test_prepare_message_and_ctx_handles_none_context() -> None:
    msg, ctx = Researcher._prepare_message_and_ctx(None, "hello")
    assert msg == "hello"
    assert ctx == {}

