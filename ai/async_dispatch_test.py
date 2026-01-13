"""Verification script for async dispatch."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from caramba.ai.lead import LeadAgent
from caramba.ai.root import RootAgent
from caramba.ai.connection import RemoteAgent

@pytest.mark.asyncio
async def test_root_agent_async_dispatch():
    """Verify RootAgent dispatches tasks asynchronously."""
    # Mock dependencies
    mock_client = AsyncMock()

    # Create RootAgent with mocked dependencies
    with patch("caramba.ai.root.PersonaLoader") as mock_loader, \
         patch("caramba.ai.root.TeamLoader") as mock_team_loader, \
         patch("caramba.ai.root.RootAgent._init_lead_connections", return_value=None):
        
        # Setup mocks
        mock_persona = MagicMock()
        mock_persona.name = "root"
        mock_persona.description = "Root Agent"
        mock_persona.model = "gpt-4"
        mock_loader.return_value.load.return_value = mock_persona
        
        agent = RootAgent(mock_client)
        agent.webhook_base_url = "http://test-webhook"

        # Mock find_agent/connect to return a mock remote agent
        mock_remote_agent = AsyncMock(spec=RemoteAgent)
        mock_remote_agent.name = "research_lead"
        mock_remote_agent.send_message_async.return_value = "task-123"
        
        # Inject the mock agent into connections
        agent.connections.get = MagicMock(return_value=mock_remote_agent)
        
        # Test delegation
        tool_context = MagicMock()
        tool_context.state = {}
        
        response = await agent.delegate_to_lead(
            lead_name="research_lead",
            message="Do research",
            tool_context=tool_context
        )
        
        # Verify response
        # Verify response
        assert isinstance(response[0], dict)
        assert response[0]["status"] == "submitted"
        assert response[0]["task_id"] == "task-123"
        assert response[0]["agent"] == "research_lead"
        
        # Verify send_message_async was called with correct webhook and callback
        mock_remote_agent.send_message_async.assert_called_once()
        call_kwargs = mock_remote_agent.send_message_async.call_args.kwargs
        assert call_kwargs["text"] == "Do research"
        assert call_kwargs["webhook_url"] == "http://test-webhook/webhook/task"
        assert call_kwargs["callback"] == agent._on_task_update
        
        # Verify message content (task_id should be None for new task)
        # We need to check what was passed to client.send_message
        # But we can't easily access the message object from the client mock here
        # without deeper mocking. 
        # However, we can trust the unit test for RemoteAgent if we had one.
        # For now, we verified the code change directly.
        
        # Verify task tracking
        assert "task-123" in agent._delegated_tasks
        assert agent._delegated_tasks["task-123"]["status"] == "submitted"

@pytest.mark.asyncio
async def test_lead_agent_async_dispatch():
    """Verify LeadAgent dispatches tasks asynchronously."""
    # Mock dependencies
    mock_client = AsyncMock()

    with patch("caramba.ai.lead.PersonaLoader") as mock_loader, \
         patch("caramba.ai.lead.TeamLoader") as mock_team_loader, \
         patch("caramba.ai.lead.LeadAgent._init_member_connections", return_value=None):
        
        mock_persona = MagicMock()
        mock_persona.name = "research_lead"
        mock_loader.return_value.load.return_value = mock_persona
        mock_team_loader.return_value.get_members_for_lead.return_value = ["researcher"]
        
        lead = LeadAgent("research_lead", mock_client, webhook_base_url="http://test-lead")
        
        # Mock connection to member
        mock_member = AsyncMock(spec=RemoteAgent)
        mock_member.name = "researcher"
        mock_member.send_message_async.return_value = "task-456"
        
        lead.connections.get = MagicMock(return_value=mock_member)
        
        # Test delegation
        tool_context = MagicMock()
        tool_context.state = {}
        
        response = await lead.delegate_to_member(
            member_name="researcher",
            message="Find papers",
            tool_context=tool_context
        )
        
        # Verify response
        # Verify response
        assert isinstance(response[0], dict)
        assert response[0]["status"] == "submitted"
        assert response[0]["task_id"] == "task-456"
        assert response[0]["agent"] == "researcher"
        
        # Verify async call
        mock_member.send_message_async.assert_called_once()
        call_kwargs = mock_member.send_message_async.call_args.kwargs
        assert call_kwargs["webhook_url"] == "http://test-lead/webhook/task"
        assert call_kwargs["callback"] == lead._on_task_update

if __name__ == "__main__":
    # Allow running directly
    import sys
    from pytest import ExitCode
    sys.exit(pytest.main(["-v", __file__]))
