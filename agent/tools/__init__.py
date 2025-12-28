"""Agent tools via MCP servers.

Tools are provided via MCP servers - the agent SDK handles discovery and execution.

Available MCP servers:
- graphiti-mcp (external): Graph memory at http://localhost:8000/mcp
- paper-tools (local): Paper writing tools, run via `python -m agent.tools.paper`

Usage:
    from agent.tools import PaperTool, GraphTool

    paper = PaperTool()
    graph = GraphTool()

    async with paper.get_server() as paper_server, graph.get_server() as graph_server:
        agent = Agent(
            name="Research Assistant",
            mcp_servers=[paper_server, graph_server],
        )
"""
from __future__ import annotations

from agents.mcp import MCPServerStdio

from agent.tools.paper import PaperTool
from agent.tools.deeplake import DeepLakeTool

TOOLS = {
    "paper": PaperTool(),
    "deeplake": DeepLakeTool(),
}


class Tool:
    """Generic tool class."""

    def __init__(self, name: str):
        self.mcp = TOOLS[name]

    def __call__(self) -> MCPServerStdio:
        """Get the MCP client for the tool.

        Returns:
            MCP client for the tool.
        """
        return self.mcp_client()

    def mcp_client(self) -> MCPServerStdio:
        """Create an MCP stdio client for this tool.

        The tool instance must provide:
        - get_command() -> str
        - get_args() -> list[str]
        """
        return MCPServerStdio(
            cache_tools_list=True,
            params={
                "command": self.mcp.get_command(),
                "args": self.mcp.get_args(),
            },
        )