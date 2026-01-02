"""Filesystem tool

Run this as a standalone server:
    python -m ai.tools.filesystem.tool

Then connect to it via MCP SSE at http://localhost:8001/sse
"""

from __future__ import annotations

import argparse
import os
import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Filesystem Tool", json_response=True)

@mcp.tool()
def list_directory(path: str) -> list[str]:
    """List the contents of a directory"""
    return os.listdir(path)

@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a file"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@mcp.tool()
def search_text(path: str, text: str) -> list[str]:
    """Search the contents of a directory recursively for a given text"""
    results = []
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and common binary/cache folders
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv", "venv"}]
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    if text in f.read():
                        results.append(file_path)
            except (UnicodeDecodeError, PermissionError, OSError):
                # Skip binary files or files we can't read
                continue
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_SERVER_PORT", "8001")))
    parser.add_argument("--host", type=str, default=os.getenv("MCP_SERVER_HOST", "0.0.0.0"))
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    app = mcp.sse_app()

    def root(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    app.add_route("/", root, methods=["GET"])
    app.add_route("/health", health, methods=["GET"])

    uvicorn.run(app, host=args.host, port=args.port)
