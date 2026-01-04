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
    if not path:
        raise ValueError("path must be a non-empty string")
    return os.listdir(path)

@mcp.tool()
def read_file(
    path: str,
    *,
    current_context_tokens: int = 0,
    context_window_tokens: int = 128000,
    max_response_tokens: int = 8192,
    mode: str = "truncate",
) -> str:
    """Read a file with context-budget safety.

    This tool prevents accidental context overflow by:
    - estimating the token footprint of the requested read
    - returning either a warning (mode="warn") or a truncated excerpt (mode="truncate")
    - exposing `read_file_lines()` so an agent can page through large files
    """
    if not path:
        raise ValueError("path must be a non-empty string")
    if current_context_tokens < 0:
        raise ValueError("current_context_tokens must be >= 0")
    if context_window_tokens <= 0:
        raise ValueError("context_window_tokens must be > 0")
    if max_response_tokens <= 0:
        raise ValueError("max_response_tokens must be > 0")
    if mode not in {"warn", "truncate"}:
        raise ValueError("mode must be 'warn' or 'truncate'")

    stat = os.stat(path)
    estimated_file_tokens = max(1, stat.st_size // 3)
    remaining_tokens = max(0, context_window_tokens - current_context_tokens)
    safety_margin = 2048
    allowed_tokens = max(0, min(max_response_tokens, remaining_tokens - safety_margin))

    if allowed_tokens <= 0:
        return (
            "File read refused: would overflow the current shared context.\n\n"
            f"- path: {path}\n"
            f"- size_bytes: {stat.st_size}\n"
            f"- estimated_file_tokens: {estimated_file_tokens}\n"
            f"- current_context_tokens: {current_context_tokens}\n"
            f"- context_window_tokens: {context_window_tokens}\n"
            f"- max_response_tokens: {max_response_tokens}\n\n"
            "Use `read_file_lines(path, start_line=..., max_lines=...)` to page through the file."
        )

    if mode == "warn" and estimated_file_tokens > allowed_tokens:
        return (
            "File read refused: would overflow the current shared context.\n\n"
            f"- path: {path}\n"
            f"- size_bytes: {stat.st_size}\n"
            f"- estimated_file_tokens: {estimated_file_tokens}\n"
            f"- allowed_response_tokens: {allowed_tokens}\n"
            f"- current_context_tokens: {current_context_tokens}\n"
            f"- context_window_tokens: {context_window_tokens}\n\n"
            "Switch to mode='truncate' or use `read_file_lines()` to page through."
        )

    return read_file_lines(
        path,
        start_line=1,
        max_lines=1000000000,
        current_context_tokens=current_context_tokens,
        context_window_tokens=context_window_tokens,
        max_response_tokens=allowed_tokens,
        mode=mode,
    )


@mcp.tool()
def read_file_lines(
    path: str,
    *,
    start_line: int = 1,
    max_lines: int = 2000,
    current_context_tokens: int = 0,
    context_window_tokens: int = 128000,
    max_response_tokens: int = 8192,
    mode: str = "truncate",
) -> str:
    """Read a line window from a file, safely.

    Agents can call this repeatedly to page through large files without blowing
    the shared context.
    """
    if not path:
        raise ValueError("path must be a non-empty string")
    if start_line <= 0:
        raise ValueError("start_line must be >= 1")
    if max_lines <= 0:
        raise ValueError("max_lines must be > 0")
    if current_context_tokens < 0:
        raise ValueError("current_context_tokens must be >= 0")
    if context_window_tokens <= 0:
        raise ValueError("context_window_tokens must be > 0")
    if max_response_tokens <= 0:
        raise ValueError("max_response_tokens must be > 0")
    if mode not in {"warn", "truncate"}:
        raise ValueError("mode must be 'warn' or 'truncate'")

    remaining_tokens = max(0, context_window_tokens - current_context_tokens)
    safety_margin = 2048
    allowed_tokens = max(0, min(max_response_tokens, remaining_tokens - safety_margin))
    if allowed_tokens <= 0:
        return (
            "File read refused: would overflow the current shared context.\n\n"
            f"- path: {path}\n"
            f"- start_line: {start_line}\n"
            f"- max_lines: {max_lines}\n"
            f"- current_context_tokens: {current_context_tokens}\n"
            f"- context_window_tokens: {context_window_tokens}\n"
            f"- max_response_tokens: {max_response_tokens}\n"
        )

    char_budget = allowed_tokens * 3
    lines_out: list[str] = []
    returned = 0
    current_line = 0
    last_line_included = start_line - 1

    with open(path, encoding="utf-8") as f:
        for line in f:
            current_line += 1
            if current_line < start_line:
                continue
            if returned >= max_lines:
                break

            if sum(len(x) for x in lines_out) + len(line) > char_budget:
                break
            lines_out.append(line)
            returned += 1
            last_line_included = current_line

        more_lines_available = bool(f.readline())

    if mode == "warn" and more_lines_available:
        return (
            "File read refused: would overflow the current shared context.\n\n"
            f"- path: {path}\n"
            f"- start_line: {start_line}\n"
            f"- max_lines: {max_lines}\n"
            f"- allowed_response_tokens: {allowed_tokens}\n\n"
            "Switch to mode='truncate' to receive a partial excerpt."
        )

    header = (
        "FILE READ (safe)\n"
        f"- path: {path}\n"
        f"- requested_start_line: {start_line}\n"
        f"- returned_start_line: {start_line if returned else None}\n"
        f"- returned_end_line: {last_line_included if returned else None}\n"
        f"- returned_lines: {returned}\n"
        f"- truncated: {bool(more_lines_available)}\n"
    )
    if more_lines_available:
        header += f"- next_start_line: {last_line_included + 1}\n"

    return header + "\n---\n" + "".join(lines_out)

@mcp.tool()
def search_text(path: str, text: str) -> list[str]:
    """Search the contents of a directory recursively for a given text"""
    if not path:
        raise ValueError("path must be a non-empty string")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    results = []
    for root, dirs, files in os.walk(path):
        # Skip hidden directories and common binary/cache folders
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv", "venv"}]

        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if text in line:
                        results.append(file_path)
                        break
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
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
